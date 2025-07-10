#!/usr/bin/env python3
"""
CKKS Linear Transform Lowering Pass with xDSL

This pass lowers ckks.linear_transform operations to explicit BSGS 
(Baby Step Giant Step) operations using xDSL for robust MLIR manipulation.

Usage:
    python ckks-linear-transform-lowering.py input.mlir -o output.mlir [--lower-bsgs]

Options:
    --lower-bsgs    Lower linear transforms into explicit BSGS operations
    --help          Show this help message
"""

import argparse
import sys
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from xdsl.ir import Operation, Region, Block, SSAValue
from xdsl.dialects import builtin, arith, func
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, FloatAttr, DenseIntOrFPElementsAttr, TensorType, f32, i32
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.context import Context
from xdsl.pattern_rewriter import (
    PatternRewriter, 
    RewritePattern, 
    PatternRewriteWalker,
    op_type_rewrite_pattern
)

# Import HEIR dialects from your project
from orion_heir.dialects.ckks import CKKS, LinearTransformOp, RotateOp, AddOp, MulPlainOp
from orion_heir.dialects.lwe import LWE, RLWEEncodeOp
from orion_heir.dialects.rns import RNS
from orion_heir.dialects.mod_arith import ModArith

@dataclass
class BSGSParameters:
    """BSGS algorithm parameters calculated from metadata."""
    diagonal_count: int
    slots: int
    bsgs_ratio: float
    baby_step_size: int
    giant_step_size: int
    num_giant_steps: int
    orion_level: int
    
    @classmethod
    def from_linear_transform_op(cls, op: LinearTransformOp) -> "BSGSParameters":
        """Extract BSGS parameters from linear transform operation attributes."""
        
        # Extract attributes
        diagonal_count = int(op.attributes.get("diagonal_count", IntegerAttr(128, i32)).value.data)
        slots = int(op.attributes.get("slots", IntegerAttr(4096, i32)).value.data)
        bsgs_ratio = float(op.attributes.get("bsgs_ratio", FloatAttr(2.0, f32)).value.data)
        orion_level = int(op.attributes.get("orion_level", IntegerAttr(5, i32)).value.data)
        
        # Calculate BSGS parameters following Lattigo's approach
        sqrt_slots = int(math.sqrt(slots))
        baby_step_size = max(1, int(sqrt_slots / bsgs_ratio))
        giant_step_size = slots // baby_step_size if baby_step_size > 0 else slots
        num_giant_steps = math.ceil(diagonal_count / baby_step_size)
        
        return cls(
            diagonal_count=diagonal_count,
            slots=slots,
            bsgs_ratio=bsgs_ratio,
            baby_step_size=baby_step_size,
            giant_step_size=giant_step_size,
            num_giant_steps=num_giant_steps,
            orion_level=orion_level
        )


class LinearTransformBSGSLoweringPattern(RewritePattern):
    """Pattern to lower ckks.linear_transform to explicit BSGS operations."""
    
    def __init__(self):
        super().__init__()
        self.var_counter = 0
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LinearTransformOp, rewriter: PatternRewriter):
        """Match linear transform operations and rewrite to BSGS."""
        
        # Extract BSGS parameters from operation metadata
        try:
            bsgs_params = BSGSParameters.from_linear_transform_op(op)
        except Exception as e:
            print(f"Failed to extract BSGS parameters: {e}")
            return
        
        print(f"🔧 Lowering linear transform with BSGS:")
        print(f"    Diagonals: {bsgs_params.diagonal_count}")
        print(f"    Baby step size: {bsgs_params.baby_step_size}")
        print(f"    Giant steps: {bsgs_params.num_giant_steps}")
        
        # Get operands
        input_ciphertext = op.operands[0]
        weights_plaintext = op.operands[1] if len(op.operands) > 1 else None
        
        if weights_plaintext is None:
            print("⚠️  No weights plaintext found, skipping lowering")
            return
        
        # Generate BSGS lowering
        result = self._generate_bsgs_operations(
            input_ciphertext, weights_plaintext, bsgs_params, rewriter
        )
        
        # Replace the original operation
        rewriter.replace_matched_op(result)
    
    def _generate_bsgs_operations(self, 
                                input_ct: SSAValue, 
                                weights_pt: SSAValue,
                                params: BSGSParameters,
                                rewriter: PatternRewriter) -> SSAValue:
        """Generate the complete BSGS operation sequence."""
        
        # Step 1: Pre-compute baby step rotations
        baby_rotations = self._generate_baby_step_rotations(
            input_ct, params.baby_step_size, rewriter
        )
        
        # Step 2: Process each giant step
        giant_step_results = []
        
        for giant_step in range(params.num_giant_steps):
            giant_result = self._process_giant_step(
                giant_step, input_ct, weights_pt, baby_rotations, params, rewriter
            )
            giant_step_results.append(giant_result)
        
        # Step 3: Accumulate all giant step results
        final_result = self._accumulate_results(giant_step_results, rewriter)
        
        return final_result
    
    def _generate_baby_step_rotations(self, 
                                    input_ct: SSAValue,
                                    baby_step_size: int,
                                    rewriter: PatternRewriter) -> List[SSAValue]:
        """Pre-compute rotations for baby steps (0, 1, 2, ..., baby_step_size-1)."""
        
        baby_rotations = []
        
        # Rotation 0 is just the original input
        baby_rotations.append(input_ct)
        
        # Generate rotations 1, 2, ..., baby_step_size-1
        for i in range(1, baby_step_size):
            rotation_op = RotateOp.build(
                operands=[input_ct],
                result_types=[input_ct.type],
                attributes={"offset": IntegerAttr(i, i32)}
            )
            rewriter.insert_op_before_matched_op(rotation_op)
            baby_rotations.append(rotation_op.results[0])
        
        return baby_rotations
    
    def _process_giant_step(self,
                          giant_step: int,
                          input_ct: SSAValue,
                          weights_pt: SSAValue, 
                          baby_rotations: List[SSAValue],
                          params: BSGSParameters,
                          rewriter: PatternRewriter) -> SSAValue:
        """Process one giant step: handle diagonals [giant_step * baby_step_size : (giant_step+1) * baby_step_size]."""
        
        start_diag = giant_step * params.baby_step_size
        end_diag = min(start_diag + params.baby_step_size, params.diagonal_count)
        num_diags_in_step = end_diag - start_diag
        
        # Apply giant step rotation (except for first giant step)
        if giant_step == 0:
            giant_input = input_ct
        else:
            giant_offset = giant_step * params.baby_step_size
            giant_rotation_op = RotateOp.build(
                operands=[input_ct],
                result_types=[input_ct.type],
                attributes={"offset": IntegerAttr(giant_offset, i32)}
            )
            rewriter.insert_op_before_matched_op(giant_rotation_op)
            giant_input = giant_rotation_op.results[0]
        
        # Process each baby step within this giant step
        baby_results = []
        
        for baby_step in range(num_diags_in_step):
            diag_idx = start_diag + baby_step
            
            # Extract diagonal from weights tensor
            diagonal_pt = self._extract_diagonal(
                weights_pt, diag_idx, params.slots, rewriter
            )
            
            # Get the appropriately rotated input
            if giant_step == 0:
                # Use pre-computed baby step rotation
                rotated_input = baby_rotations[baby_step]
            else:
                # Apply baby step rotation to giant-step-rotated input
                baby_rotation_op = RotateOp.build(
                    operands=[giant_input],
                    result_types=[giant_input.type],
                    attributes={"offset": IntegerAttr(baby_step, i32)}
                )
                rewriter.insert_op_before_matched_op(baby_rotation_op)
                rotated_input = baby_rotation_op.results[0]
            
            # Multiply diagonal with rotated ciphertext
            mult_op = MulPlainOp.build(
                operands=[rotated_input, diagonal_pt],
                result_types=[rotated_input.type]
            )
            rewriter.insert_op_before_matched_op(mult_op)
            baby_results.append(mult_op.results[0])
        
        # Accumulate baby step results within this giant step
        return self._accumulate_results(baby_results, rewriter)
    
    def _extract_diagonal(self,
                         weights_tensor: SSAValue,
                         diag_idx: int,
                         slots: int,
                         rewriter: PatternRewriter) -> SSAValue:
        """Extract a single diagonal from the weights tensor."""
        
        # TODO: Implement proper tensor slicing to extract diagonal diag_idx
        # For now, create a placeholder constant with the right shape
        
        diagonal_data = [0.0] * slots  # Placeholder - should extract real data
        
        diagonal_type = TensorType(f32, [slots])
        diagonal_attr = DenseIntOrFPElementsAttr.create_dense_float(
            diagonal_type, diagonal_data
        )
        
        const_op = arith.ConstantOp.build(
            attributes={"value": diagonal_attr},
            result_types=[diagonal_type]
        )
        rewriter.insert_op_before_matched_op(const_op)
        
        # Get the encoding and ring attributes from the original weights tensor
        # We need to match the original RLWEEncodeOp that created weights_tensor
        original_encoding = None
        original_ring = None
        
        # Try to find the defining op of weights_tensor to get its attributes
        if hasattr(weights_tensor, 'owner') and weights_tensor.owner:
            defining_op = weights_tensor.owner
            if hasattr(defining_op, 'encoding'):
                original_encoding = defining_op.encoding
            if hasattr(defining_op, 'ring'):
                original_ring = defining_op.ring
        
        # If we can't find the original attributes, create reasonable defaults
        if original_encoding is None:
            # Create a default InverseCanonicalEncodingAttr
            from orion_heir.dialects.lwe import InverseCanonicalEncodingAttr
            original_encoding = InverseCanonicalEncodingAttr([IntegerAttr(45, i32)])  # scaling_factor = 45
        
        if original_ring is None:
            # Create a default RingAttr  
            from orion_heir.dialects.lwe import RingAttr
            # This should match your scheme parameters
            original_ring = RingAttr([
                f32,  # coefficientType
                # polynomial modulus and other ring parameters would go here
            ])
        
        # Create the result type - use the same type as the original weights
        result_type = weights_tensor.type
        
        # Encode to plaintext with proper attributes
        encode_op = RLWEEncodeOp.build(
            operands=[const_op.results[0]],
            result_types=[result_type],
            attributes={
                "encoding": original_encoding,
                "ring": original_ring
            }
        )
        rewriter.insert_op_before_matched_op(encode_op)
        
        return encode_op.results[0]
    
    def _accumulate_results(self, results: List[SSAValue], rewriter: PatternRewriter) -> SSAValue:
        """Accumulate a list of results using addition operations."""
        
        if len(results) == 0:
            raise ValueError("Cannot accumulate empty results list")
        
        if len(results) == 1:
            return results[0]
        
        # Accumulate pairwise: ((a + b) + c) + d + ...
        current_result = results[0]
        
        for i in range(1, len(results)):
            add_op = AddOp.build(
                operands=[current_result, results[i]],
                result_types=[current_result.type]
            )
            rewriter.insert_op_before_matched_op(add_op)
            current_result = add_op.results[0]
        
        return current_result


class CKKSLinearTransformLoweringPass:
    """Main pass for lowering CKKS linear transforms."""
    
    def __init__(self, lower_bsgs: bool = False):
        self.lower_bsgs = lower_bsgs
        self.context = Context()
        self._register_dialects()
    
    def _register_dialects(self):
        """Register required dialects."""
        # Register builtin dialects
        self.context.load_dialect(builtin.Builtin)
        self.context.load_dialect(arith.Arith)
        self.context.load_dialect(func.Func)
        self.context.load_dialect(CKKS)
        self.context.load_dialect(LWE)
        self.context.load_dialect(RNS)
        self.context.load_dialect(ModArith)
    
    def run_pass(self, module: ModuleOp) -> ModuleOp:
        """Run the linear transform lowering pass."""
        
        if not self.lower_bsgs:
            print("ℹ️  BSGS lowering disabled, returning original module")
            return module
        
        print("🚀 Running CKKS Linear Transform BSGS Lowering Pass")
        
        # Create pattern rewriter
        pattern = LinearTransformBSGSLoweringPattern()
        walker = PatternRewriteWalker(pattern)
        
        # Apply the pattern
        walker.rewrite_module(module)
        
        print("✅ BSGS lowering completed")
        return module
    
    def process_file(self, input_file: str, output_file: str):
        """Process an MLIR file."""
        
        # Parse input
        if input_file == "-":
            input_text = sys.stdin.read()
        else:
            with open(input_file, 'r') as f:
                input_text = f.read()
        
        # Parse MLIR
        parser = Parser(self.context, input_text)
        module = parser.parse_module()
        
        # Run the pass
        transformed_module = self.run_pass(module)
        
        # Write output
        printer = Printer()
        output_text = printer.print_to_string(transformed_module)
        
        if output_file == "-":
            print(output_text)
        else:
            with open(output_file, 'w') as f:
                f.write(output_text)


def main():
    parser = argparse.ArgumentParser(
        description="Lower CKKS linear transform operations using xDSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Lower linear transforms to explicit BSGS operations
    python ckks-linear-transform-lowering.py input.mlir -o output.mlir --lower-bsgs
    
    # Process stdin to stdout
    cat input.mlir | python ckks-linear-transform-lowering.py --lower-bsgs > output.mlir
        """
    )
    
    parser.add_argument('input', nargs='?', default='-',
                        help='Input MLIR file (default: stdin)')
    parser.add_argument('-o', '--output', default='-',
                        help='Output MLIR file (default: stdout)')
    parser.add_argument('--lower-bsgs', action='store_true',
                        help='Lower linear transforms into explicit BSGS operations')
    
    args = parser.parse_args()
    
    # Create and run the pass
    pass_instance = CKKSLinearTransformLoweringPass(lower_bsgs=args.lower_bsgs)
    
    try:
        pass_instance.process_file(args.input, args.output)
        if args.lower_bsgs:
            print("✅ CKKS Linear Transform lowering completed", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error during lowering: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
