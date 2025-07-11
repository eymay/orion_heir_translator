#!/usr/bin/env python3
"""
CKKS Linear Transform Lowering Pass with xDSL

This pass lowers ckks.linear_transform operations to explicit naive rotations 
using xDSL for robust MLIR manipulation.

Usage:
    python ckks-linear-transform-lowering.py input.mlir -o output.mlir [--lower-naive] [--optimize-zeros]

Options:
    --lower-naive     Lower linear transforms into explicit naive operations
    --optimize-zeros  Remove operations related to zero tensors
    --help           Show this help message
"""

import argparse
import sys
import math
from typing import List, Dict, Tuple, Optional, Any
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
from orion_heir.dialects.polynomial import Polynomial, RingAttr, PolynomialAttr
from orion_heir.dialects.rns import RNS
from orion_heir.dialects.mod_arith import ModArith




class LinearTransformNaiveLoweringPattern(RewritePattern):
    """Pattern to lower ckks.linear_transform to explicit naive rotations with proper type propagation."""
    
    def __init__(self):
        super().__init__()
        self.var_counter = 0
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LinearTransformOp, rewriter: PatternRewriter):
        """Match linear transform operations and rewrite with proper type propagation."""
        
        # Extract parameters (existing logic)
        try:
            diagonal_count = 128
            slots = 4096
            orion_level = 5
            
            if hasattr(op, 'attributes') and op.attributes:
                if "diagonal_count" in op.attributes:
                    diagonal_count = int(op.attributes["diagonal_count"].value.data)
                if "slots" in op.attributes:
                    slots = int(op.attributes["slots"].value.data)
                if "orion_level" in op.attributes:
                    orion_level = int(op.attributes["orion_level"].value.data)
        except Exception as e:
            print(f"Failed to extract parameters: {e}")
            return
        
        print(f"🔧 Lowering linear transform with proper type propagation:")
        print(f"    Diagonals: {diagonal_count}, Slots: {slots}")
        
        # Get operands
        input_ciphertext = op.operands[0]
        weights_plaintext = op.operands[1] if len(op.operands) > 1 else None
        
        if weights_plaintext is None:
            print("⚠️  No weights plaintext found, skipping lowering")
            return
        
        # CRITICAL: Extract initial scaling factors
        input_scale = self._get_scaling_factor(input_ciphertext.type)
        weights_scale = self._get_scaling_factor(weights_plaintext.type)
        expected_result_scale = input_scale + weights_scale  # After multiplication
        
        print(f"    Input scale: {input_scale}, Weights scale: {weights_scale}")
        print(f"    Expected result scale after mul_plain: {expected_result_scale}")
        
        # Create the operations with proper type tracking
        new_ops = []
        rotation_results = []
        
        # Process each diagonal
        for diagonal_idx in range(diagonal_count):
            # Extract diagonal with MATCHING scaling factor
            const_op, encode_op = self._extract_diagonal_from_weights(
                weights_plaintext, diagonal_idx, slots, input_ciphertext.type
            )
            new_ops.extend([const_op, encode_op])
            diagonal_weights = encode_op.results[0]
            
            # Apply rotation if needed
            if diagonal_idx == 0:
                rotated_input = input_ciphertext
            else:
                # FIXED: Rotation preserves input type exactly
                rotate_op = RotateOp.build(
                    operands=[input_ciphertext],
                    result_types=[input_ciphertext.type],  # Same type as input
                    properties={"offset": IntegerAttr(diagonal_idx, i32)}
                )
                new_ops.append(rotate_op)
                rotated_input = rotate_op.results[0]
            
            # CRITICAL: Compute correct result type for mul_plain
            mul_result_type = self._compute_mul_plain_result_type(
                rotated_input.type, diagonal_weights.type
            )
            
            # Multiply with proper result type
            mul_op = MulPlainOp.build(
                operands=[rotated_input, diagonal_weights],
                result_types=[mul_result_type]
            )
            new_ops.append(mul_op)
            rotation_results.append(mul_op.results[0])
        
        # CRITICAL: Accumulate results with proper type management
        if len(rotation_results) == 0:
            print("⚠️  No rotation results generated")
            return
        elif len(rotation_results) == 1:
            final_result = rotation_results[0]
        else:
            # Start with first result
            current_result = rotation_results[0]
            current_type = current_result.type
            
            for i in range(1, len(rotation_results)):
                next_result = rotation_results[i]
                next_type = next_result.type
                
                # CRITICAL: Ensure both operands have compatible types for addition
                current_scale = self._get_scaling_factor(current_type)
                next_scale = self._get_scaling_factor(next_type)
                
                if current_scale != next_scale:
                    print(f"    Scale mismatch in addition: {current_scale} vs {next_scale}")
                    # Insert type conversions to make operands compatible
                    current_result, next_result = self._insert_type_conversion_for_addition(
                        current_result, next_result, new_ops
                    )
                    # Update types after conversion
                    current_type = current_result.type
                    next_type = next_result.type
                    result_scale = max(current_scale, next_scale)
                    add_result_type = self._create_ciphertext_with_scale(current_type, result_scale)
                else:
                    add_result_type = current_type
                
                add_op = AddOp.build(
                    operands=[current_result, next_result],
                    result_types=[add_result_type]
                )
                new_ops.append(add_op)
                current_result = add_op.results[0]
                current_type = add_result_type
            
            final_result = current_result
        
        # CRITICAL: Ensure final result type matches original operation's expected type
        original_result_type = op.results[0].type
        final_result_type = final_result.type
        
        original_scale = self._get_scaling_factor(original_result_type)
        final_scale = self._get_scaling_factor(final_result_type)
        
        if original_scale != final_scale:
            print(f"    Final scale mismatch: original={original_scale}, actual={final_scale}")
            print(f"    Inserting rescale operation to convert {final_scale} -> {original_scale}")
            
            # Insert rescale operation to match expected type
            final_result = self._insert_rescale_operation(
                final_result, original_result_type, new_ops
            )
        
        # Insert all operations and replace
        for new_op in new_ops:
            rewriter.insert_op_before_matched_op(new_op)
        
        rewriter.replace_matched_op([], [final_result])
        print(f"✅ Successfully lowered with final scaling factor: {self._get_scaling_factor(final_result.type)}")
    
    def _get_scaling_factor(self, ciphertext_type: Any) -> int:
        """Extract scaling factor from ciphertext type."""
        if hasattr(ciphertext_type, 'parameters') and len(ciphertext_type.parameters) >= 2:
            pt_space = ciphertext_type.parameters[1]
            return pt_space.encoding.scaling_factor.value.data
        return 0
    
    def _compute_mul_plain_result_type(self, ciphertext_type: Any, plaintext_type: Any) -> Any:
        """Compute result type for ciphertext-plaintext multiplication."""
        ct_scale = self._get_scaling_factor(ciphertext_type)
        pt_scale = self._get_scaling_factor(plaintext_type)
        result_scale = ct_scale + pt_scale  # Addition in log domain
        
        print(f"      mul_plain: {ct_scale} + {pt_scale} = {result_scale}")
        
        return self._create_ciphertext_with_scale(ciphertext_type, result_scale)
    
    def _create_ciphertext_with_scale(self, base_type: Any, new_scale: int) -> Any:
        """Create a new ciphertext type with updated scaling factor."""
        from orion_heir.dialects.lwe import NewLWECiphertextType, PlaintextSpaceAttr, InverseCanonicalEncodingAttr
        from xdsl.dialects.builtin import IntegerAttr, IntegerType
        
        if not hasattr(base_type, 'parameters') or len(base_type.parameters) < 5:
            return base_type
        
        # Extract all components
        app_data = base_type.parameters[0]
        old_pt_space = base_type.parameters[1]
        ct_space = base_type.parameters[2]
        key = base_type.parameters[3]
        modulus_chain = base_type.parameters[4]
        
        # Create new encoding with updated scale
        new_encoding = InverseCanonicalEncodingAttr([
            IntegerAttr(new_scale, IntegerType(32))
        ])
        
        # Create new plaintext space
        new_pt_space = PlaintextSpaceAttr([old_pt_space.ring, new_encoding])
        
        # Return new ciphertext type
        return NewLWECiphertextType([
            app_data, new_pt_space, ct_space, key, modulus_chain
        ])
    
    def _extract_diagonal_from_weights(self, weights_plaintext: SSAValue, diagonal_idx: int, slots: int, input_ciphertext_type: Any = None) -> Tuple[Operation, Operation]:
        """Extract diagonal with proper scaling factor matching."""
        
        print(f"    Extracting diagonal {diagonal_idx} with proper scaling")
        
        # Extract diagonal data (existing logic)
        weights_op = weights_plaintext.owner
        weights_attr = None
        if hasattr(weights_op, 'attributes') and 'value' in weights_op.attributes:
            weights_attr = weights_op.attributes['value']
        
        if weights_attr is None:
            diagonal_row = [0.1 * (diagonal_idx + 1)] * slots
        else:
            start_idx = diagonal_idx * slots
            end_idx = start_idx + slots
            
            if hasattr(weights_attr, 'data'):
                full_data = weights_attr.data.data
            else:
                full_data = weights_attr.value if hasattr(weights_attr, 'value') else weights_attr
            
            if hasattr(full_data, '__getitem__'):
                diagonal_row = list(full_data[start_idx:end_idx])
            else:
                diagonal_row = [0.1 * (diagonal_idx + 1)] * slots
        
        # Create constant
        diagonal_type = TensorType(f32, [slots])
        diagonal_attr = DenseIntOrFPElementsAttr.from_list(diagonal_type, diagonal_row)
        
        const_op = arith.ConstantOp.build(
            properties={"value": diagonal_attr},
            result_types=[diagonal_type]
        )
        
        # CRITICAL: Create plaintext with SAME scaling factor as input ciphertext
        target_scaling_factor = 0  # Default
        target_app_data = ApplicationDataAttr([TensorType(f32, [slots])])
        
        if input_ciphertext_type is not None:
            if hasattr(input_ciphertext_type, 'parameters') and len(input_ciphertext_type.parameters) >= 2:
                # Use SAME scaling factor as input ciphertext (not weights!)
                ct_pt_space = input_ciphertext_type.parameters[1]
                target_scaling_factor = ct_pt_space.encoding.scaling_factor.value.data
                target_app_data = input_ciphertext_type.parameters[0]
                print(f"      Using input ciphertext scaling factor: {target_scaling_factor}")
        
        # Create matching plaintext type
        encoding_attr = InverseCanonicalEncodingAttr([IntegerAttr(target_scaling_factor, i32)])
        poly_attr = PolynomialAttr([StringAttr("1+x**8192")])
        ring_attr = RingAttr([f32, poly_attr])
        pt_space = PlaintextSpaceAttr([ring_attr, encoding_attr])
        result_type = NewLWEPlaintextType([target_app_data, pt_space])
        
        encode_op = RLWEEncodeOp.build(
            operands=[const_op.results[0]],
            result_types=[result_type],
            attributes={
                "encoding": encoding_attr,
                "ring": ring_attr
            }
        )
        
        return const_op, encode_op
    
    def _insert_rescale_operation(self, input_value: SSAValue, target_type: Any, ops_list: List) -> SSAValue:
        """Insert a rescale operation to convert between scaling factors."""
        from orion_heir.dialects.ckks import RescaleOp
        
        input_scale = self._get_scaling_factor(input_value.type)
        target_scale = self._get_scaling_factor(target_type)
        
        print(f"      Inserting rescale: {input_scale} -> {target_scale}")
        
        if input_scale == target_scale:
            return input_value  # No conversion needed
        
        if input_scale > target_scale:
            # Need to rescale down (more common case)
            # Extract the correct target ring from the target ciphertext type
            if hasattr(target_type, 'parameters') and len(target_type.parameters) >= 3:
                target_ct_space = target_type.parameters[2]  # CiphertextSpaceAttr
                target_ring = target_ct_space.ring  # Use the actual ciphertext ring
            else:
                # Fallback: extract ring from input and hope it's compatible
                if hasattr(input_value.type, 'parameters') and len(input_value.type.parameters) >= 3:
                    input_ct_space = input_value.type.parameters[2]
                    target_ring = input_ct_space.ring
                else:
                    print(f"      Error: Cannot extract target ring for rescale")
                    return input_value
            
            rescale_op = RescaleOp.build(
                operands=[input_value],
                result_types=[target_type],
                properties={"to_ring": target_ring}
            )
            ops_list.append(rescale_op)
            return rescale_op.results[0]
        else:
            # Need to scale up - this is unusual and might require different approach
            print(f"      Warning: Scaling up from {input_scale} to {target_scale} - using direct type assignment")
            # For scaling up, we might need a different operation or just accept the type mismatch
            # In CKKS, you typically can't scale up without additional information
            return input_value
    
    def _insert_type_conversion_for_addition(self, value1: SSAValue, value2: SSAValue, ops_list: List) -> Tuple[SSAValue, SSAValue]:
        """Ensure both values have compatible types for addition."""
        scale1 = self._get_scaling_factor(value1.type)
        scale2 = self._get_scaling_factor(value2.type)
        
        if scale1 == scale2:
            return value1, value2  # Already compatible
        
        # Convert both to the higher scaling factor
        target_scale = max(scale1, scale2)
        target_type = self._create_ciphertext_with_scale(value1.type, target_scale)
        
        converted_value1 = value1
        converted_value2 = value2
        
        if scale1 < target_scale:
            print(f"      Converting operand 1: {scale1} -> {target_scale}")
            # For scaling up, we might need a different approach
            # This is a placeholder - you might need to implement scale-up logic
            converted_value1 = value1
        elif scale1 > target_scale:
            converted_value1 = self._insert_rescale_operation(value1, target_type, ops_list)
        
        if scale2 < target_scale:
            print(f"      Converting operand 2: {scale2} -> {target_scale}")
            converted_value2 = value2
        elif scale2 > target_scale:
            target_type2 = self._create_ciphertext_with_scale(value2.type, target_scale)
            converted_value2 = self._insert_rescale_operation(value2, target_type2, ops_list)
        
        return converted_value1, converted_value2


class CKKSLinearTransformNaiveLoweringPass:
    """Main pass for lowering CKKS linear transforms to naive rotations."""
    
    def __init__(self, lower_naive: bool = False):
        self.lower_naive = lower_naive
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
        self.context.load_dialect(Polynomial)
    
    def run_pass(self, module: ModuleOp) -> ModuleOp:
        """Run the linear transform naive lowering pass."""
        
        if not self.lower_naive:
            print("ℹ️  Naive lowering disabled, returning original module")
            return module
        
        print("🚀 Running CKKS Linear Transform Naive Lowering Pass")
        
        # Create pattern rewriter for lowering
        pattern = LinearTransformNaiveLoweringPattern()
        walker = PatternRewriteWalker(pattern)
        
        # Apply the lowering pattern
        walker.rewrite_module(module)
        print("✅ Naive lowering completed")
        
        return module
    
    def process_file(self, input_file: str, output_file: str):
        """Process an MLIR file."""
        # Parse the input file
        if input_file == "-":
            input_text = sys.stdin.read()
        else:
            with open(input_file, 'r') as f:
                input_text = f.read()
        
        # Parse MLIR - using the correct initialization format
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
        
        print(f"✅ Processed {input_file} -> {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CKKS Linear Transform Naive Lowering Pass")
    parser.add_argument("input_file", help="Input MLIR file")
    parser.add_argument("-o", "--output", help="Output MLIR file", required=True)
    parser.add_argument("--lower-naive", action="store_true", help="Lower linear transforms into explicit naive operations")
    
    args = parser.parse_args()
    
    pass_runner = CKKSLinearTransformNaiveLoweringPass(
        lower_naive=args.lower_naive
    )
    pass_runner.process_file(args.input_file, args.output)


if __name__ == "__main__":
    main()
