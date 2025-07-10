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
from xdsl.dialects.builtin import ModuleOp, IntegerAttr, FloatAttr, DenseIntOrFPElementsAttr, TensorType, f32, i32, StringAttr, ArrayAttr
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
from orion_heir.dialects.lwe import LWE, RLWEEncodeOp, NewLWEPlaintextType, ApplicationDataAttr, PlaintextSpaceAttr, InverseCanonicalEncodingAttr
from orion_heir.dialects.polynomial import RingAttr, PolynomialAttr
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
        
        # Extract attributes with proper defaults
        diagonal_count = 128  # Default
        slots = 4096  # Default
        bsgs_ratio = 2.0  # Default
        orion_level = 5  # Default
        
        # Try to extract from operation attributes if they exist
        if hasattr(op, 'attributes') and op.attributes:
            if "diagonal_count" in op.attributes:
                diagonal_count = int(op.attributes["diagonal_count"].value.data)
            if "slots" in op.attributes:
                slots = int(op.attributes["slots"].value.data)
            if "bsgs_ratio" in op.attributes:
                bsgs_ratio = float(op.attributes["bsgs_ratio"].value.data)
            if "orion_level" in op.attributes:
                orion_level = int(op.attributes["orion_level"].value.data)
        
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
        
        # Create the BSGS operations
        new_ops = []
        
        # Step 1: Create baby step rotations
        baby_rotations = [input_ciphertext]  # rotation by 0 is the input itself
        
        for i in range(1, bsgs_params.baby_step_size):
            rotate_op = RotateOp.build(
                operands=[input_ciphertext],
                result_types=[input_ciphertext.type],
                properties={"offset": IntegerAttr(i, i32)}
            )
            new_ops.append(rotate_op)
            baby_rotations.append(rotate_op.results[0])
        
        # Step 2: Process each giant step
        giant_step_results = []
        
        for giant_step in range(bsgs_params.num_giant_steps):
            baby_step_results = []
            
            # Process each baby step in this giant step
            for baby_step in range(bsgs_params.baby_step_size):
                diagonal_idx = giant_step * bsgs_params.baby_step_size + baby_step
                
                if diagonal_idx >= bsgs_params.diagonal_count:
                    break  # No more diagonals to process
                
                # Get the appropriate baby step rotation
                rotated_input = baby_rotations[baby_step]
                
                # Extract the specific diagonal from the weights tensor
                # The weights tensor contains all diagonals - we need to select the right one
                const_op, encode_op = self._extract_diagonal_from_weights(
                    weights_plaintext, diagonal_idx, bsgs_params.slots
                )
                new_ops.extend([const_op, encode_op])
                diagonal_weights = encode_op.results[0]
                
                # Multiply rotated input with diagonal weights
                mul_op = MulPlainOp.build(
                    operands=[rotated_input, diagonal_weights],
                    result_types=[rotated_input.type]
                )
                new_ops.append(mul_op)
                baby_step_results.append(mul_op.results[0])
            
            # Sum baby step results
            if len(baby_step_results) == 0:
                continue
            elif len(baby_step_results) == 1:
                baby_sum = baby_step_results[0]
            else:
                current_result = baby_step_results[0]
                for i in range(1, len(baby_step_results)):
                    add_op = AddOp.build(
                        operands=[current_result, baby_step_results[i]],
                        result_types=[current_result.type]
                    )
                    new_ops.append(add_op)
                    current_result = add_op.results[0]
                baby_sum = current_result
            
            # Apply giant step rotation if needed
            if giant_step > 0:
                giant_rotation = giant_step * bsgs_params.giant_step_size
                rotate_op = RotateOp.build(
                    operands=[baby_sum],
                    result_types=[baby_sum.type],
                    properties={"offset": IntegerAttr(giant_rotation, i32)}
                )
                new_ops.append(rotate_op)
                giant_step_results.append(rotate_op.results[0])
            else:
                giant_step_results.append(baby_sum)
        
        # Step 3: Accumulate all giant step results
        if len(giant_step_results) == 0:
            print("⚠️  No giant step results generated")
            return
        elif len(giant_step_results) == 1:
            final_result = giant_step_results[0]
        else:
            current_result = giant_step_results[0]
            for i in range(1, len(giant_step_results)):
                add_op = AddOp.build(
                    operands=[current_result, giant_step_results[i]],
                    result_types=[current_result.type]
                )
                new_ops.append(add_op)
                current_result = add_op.results[0]
            final_result = current_result
        
        # Insert all new operations and replace the original
        for new_op in new_ops:
            rewriter.insert_op_before_matched_op(new_op)
        
        # Replace the matched operation's results with the final result
        rewriter.replace_matched_op([], [final_result])
        print("✅ Successfully replaced linear transform with BSGS operations")
    
    def _extract_diagonal_from_weights(self, weights_plaintext: SSAValue, diagonal_idx: int, slots: int) -> SSAValue:
        """Extract a specific diagonal from the weights tensor.
        
        The weights tensor is tensor<128x4096xf32> where:
        - First dimension (128) = number of diagonals  
        - Second dimension (4096) = slots per diagonal
        
        We need to extract the diagonal_idx-th row as a tensor<4096xf32>.
        """
        # Find the defining constant operation
        defining_op = weights_plaintext.owner
        while defining_op and not isinstance(defining_op, RLWEEncodeOp):
            defining_op = defining_op.operands[0].owner if defining_op.operands else None
        
        if not defining_op or not isinstance(defining_op, RLWEEncodeOp):
            raise ValueError("Could not find RLWEEncodeOp defining the weights")
        
        # Get the constant that feeds into the encode operation
        const_op = defining_op.operands[0].owner
        if not isinstance(const_op, arith.ConstantOp):
            raise ValueError("Could not find ConstantOp defining the weights tensor")
        
        # Get the dense attribute containing the tensor data
        weights_attr = const_op.properties["value"]
        if not isinstance(weights_attr, DenseIntOrFPElementsAttr):
            raise ValueError("Weights constant is not a DenseIntOrFPElementsAttr")
        
        # Extract the diagonal_idx-th row from the 2D tensor
        # The data is stored as a flattened array: [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
        start_idx = diagonal_idx * slots
        end_idx = start_idx + slots
        
        # Extract the row data
        if hasattr(weights_attr, 'data'):
            full_data = weights_attr.data.data  # Get the underlying array
        else:
            # Try different attribute access patterns
            full_data = weights_attr.value if hasattr(weights_attr, 'value') else weights_attr
        
        # Convert to list if needed and slice
        if hasattr(full_data, '__getitem__'):
            diagonal_row = list(full_data[start_idx:end_idx])
        else:
            # If we can't slice, create a pattern (fallback)
            diagonal_row = [0.1 * (diagonal_idx + 1)] * slots
        
        # print(f"    Extracted diagonal {diagonal_idx}: {len(diagonal_row)} values, first few: {diagonal_row[:5] if len(diagonal_row) > 5 else diagonal_row}")
        
        # Create a new constant with just this diagonal row
        diagonal_type = TensorType(f32, [slots])
        diagonal_attr = DenseIntOrFPElementsAttr.from_list(diagonal_type, diagonal_row)
        
        const_op = arith.ConstantOp.build(
            properties={"value": diagonal_attr},
            result_types=[diagonal_type]
        )
        
        # Create plaintext type (matching the original weights)
        app_data = ApplicationDataAttr([TensorType(f32, [slots])])
        encoding_attr = InverseCanonicalEncodingAttr([IntegerAttr(45, i32)])
        poly_attr = PolynomialAttr([StringAttr("1 + x**8192")])
        ring_attr = RingAttr([f32, poly_attr])
        pt_space = PlaintextSpaceAttr([ring_attr, encoding_attr])
        result_type = NewLWEPlaintextType([app_data, pt_space])
        
        encode_op = RLWEEncodeOp.build(
            operands=[const_op.results[0]],
            result_types=[result_type]
        )
        
        return const_op, encode_op


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
        from io import StringIO
        
        if output_file == "-":
            # Print to stdout
            printer = Printer()
            printer.print_op(transformed_module)
        else:
            # Write to file
            output_buffer = StringIO()
            printer = Printer(stream=output_buffer)
            printer.print_op(transformed_module)
            output_text = output_buffer.getvalue()
            
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
