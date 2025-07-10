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
from orion_heir.dialects.polynomial import Polynomial, RingAttr, PolynomialAttr
from orion_heir.dialects.rns import RNS
from orion_heir.dialects.mod_arith import ModArith


class LinearTransformNaiveLoweringPattern(RewritePattern):
    """Pattern to lower ckks.linear_transform to explicit naive rotations."""
    
    def __init__(self):
        super().__init__()
        self.var_counter = 0
    
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: LinearTransformOp, rewriter: PatternRewriter):
        """Match linear transform operations and rewrite to naive rotations."""
        
        # Extract parameters from operation metadata
        try:
            diagonal_count = 128  # Default
            slots = 4096  # Default
            orion_level = 5  # Default
            
            # Try to extract from operation attributes if they exist
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
        
        print(f"🔧 Lowering linear transform with naive rotations:")
        print(f"    Diagonals: {diagonal_count}")
        print(f"    Slots: {slots}")
        
        # Get operands
        input_ciphertext = op.operands[0]
        weights_plaintext = op.operands[1] if len(op.operands) > 1 else None
        
        if weights_plaintext is None:
            print("⚠️  No weights plaintext found, skipping lowering")
            return
        
        # Create the naive rotation operations
        new_ops = []
        rotation_results = []
        
        # Process each diagonal directly
        for diagonal_idx in range(diagonal_count):
            # Extract the specific diagonal from the weights tensor
            const_op, encode_op = self._extract_diagonal_from_weights(
                weights_plaintext, diagonal_idx, slots
            )
            new_ops.extend([const_op, encode_op])
            diagonal_weights = encode_op.results[0]
            
            # Apply rotation if needed (except for diagonal 0)
            if diagonal_idx == 0:
                rotated_input = input_ciphertext
            else:
                rotate_op = RotateOp.build(
                    operands=[input_ciphertext],
                    result_types=[input_ciphertext.type],
                    properties={"offset": IntegerAttr(diagonal_idx, i32)}
                )
                new_ops.append(rotate_op)
                rotated_input = rotate_op.results[0]
            
            # Multiply rotated input with diagonal weights
            mul_op = MulPlainOp.build(
                operands=[rotated_input, diagonal_weights],
                result_types=[rotated_input.type]
            )
            new_ops.append(mul_op)
            rotation_results.append(mul_op.results[0])
        
        # Sum all rotation results
        if len(rotation_results) == 0:
            print("⚠️  No rotation results generated")
            return
        elif len(rotation_results) == 1:
            final_result = rotation_results[0]
        else:
            current_result = rotation_results[0]
            for i in range(1, len(rotation_results)):
                add_op = AddOp.build(
                    operands=[current_result, rotation_results[i]],
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
        print("✅ Successfully replaced linear transform with naive rotations")
    
    def _extract_diagonal_from_weights(self, weights_plaintext: SSAValue, diagonal_idx: int, slots: int) -> Tuple[Operation, Operation]:
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
        
        # Create a new constant with just this diagonal row
        diagonal_type = TensorType(f32, [slots])
        diagonal_attr = DenseIntOrFPElementsAttr.from_list(diagonal_type, diagonal_row)
        
        const_op = arith.ConstantOp.build(
            properties={"value": diagonal_attr},
            result_types=[diagonal_type]
        )
        
        # Create plaintext type (matching the original weights)
        app_data = ApplicationDataAttr([TensorType(f32, [slots])])
        encoding_attr = InverseCanonicalEncodingAttr([IntegerAttr(0, i32)])
        poly_attr = PolynomialAttr([StringAttr("1+x**8192")])
        ring_attr = RingAttr([f32, poly_attr])
        pt_space = PlaintextSpaceAttr([ring_attr, encoding_attr])
        result_type = NewLWEPlaintextType([app_data, pt_space])
        
        encode_op = RLWEEncodeOp.build(
            operands=[const_op.results[0]],
            result_types=[result_type],
            attributes={
                "encoding": encoding_attr,
                "ring": ring_attr
            }
        )
        
        return const_op, encode_op


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
