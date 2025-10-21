"""
Operation registry for modular FHE operation translation.

This module provides a registry system for different FHE operations,
allowing for easy extension and customization of translation behavior.
"""

from typing import Dict, Any, Protocol, List
from abc import ABC, abstractmethod
import torch

from xdsl.ir import SSAValue, Block
from xdsl.dialects.builtin import (
    ArrayAttr,
    DenseArrayBase,
    DenseIntOrFPElementsAttr,
    DictionaryAttr,
    FloatAttr,
    FunctionType,
    IntegerAttr,
    IntegerType,
    StringAttr,
    TensorType,
    f64,
    i32,
)
from xdsl.dialects.func import FuncOp
from xdsl.dialects.arith import ConstantOp
from ..dialects.orion import LinearTransformOp
from ..dialects.lwe import RLWEEncodeOp, LWEPlaintextType
from .translator import FHEOperation
from ..dialects.ckks import (
    AddOp,
    AddPlainOp,
    MulOp,
    MulPlainOp,
    RelinearizeOp,
    RescaleOp,
    RotateOp,
    SubOp,
)


import numpy as np


def get_constant_operand(operation: FHEOperation) -> Any:
    for arg in operation.args or []:
        if isinstance(arg, torch.Tensor):
            return arg


def get_parent_func(block: Block) -> FuncOp:
    # Constants are big, so instead write the stacked diagonal data to disk
    # and add a new func argument for it, with metadata attached.
    func_op = block.parent_op()
    while not isinstance(func_op, FuncOp):
        func_op = func_op.parent_op()
    return func_op


class OperationHandler(Protocol):
    """Protocol for operation handlers."""

    def __call__(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle the translation of an FHE operation."""
        ...


class BaseOperationHandler(ABC):
    """Base class for operation handlers."""

    @abstractmethod
    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle the translation of an FHE operation."""
        pass


class CKKSArithmeticHandler(BaseOperationHandler):
    """Handler for CKKS arithmetic operations (add, sub, mul)."""

    def __init__(self, op_class):
        self.op_class = op_class

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle CKKS arithmetic operations."""
        # Determine operands
        if operation.op_type in ["add", "sub"]:
            # Binary operations - need second operand
            second_operand = self._get_second_operand(operation, constants)
            if second_operand is None:
                # If no second operand, return current value unchanged
                return current_value

            # Determine result type
            result_type = type_builder.infer_result_type(
                operation.op_type, current_value.type, second_operand.type
            )

            # Create the operation
            op_instance = self.op_class(
                operands=[current_value, second_operand], result_types=[result_type]
            )

            block.add_op(op_instance)
            return op_instance.results[0]

        return current_value

    def _get_second_operand(
        self, operation: FHEOperation, constants: Dict[str, SSAValue]
    ) -> SSAValue:
        """Get the second operand for binary operations."""
        # Look for constants or other operands
        if operation.result_var and f"pt_{operation.result_var}_0" in constants:
            return constants[f"pt_{operation.result_var}_0"]

        # Check for @ references in args
        if operation.args:
            for arg in operation.args:
                if isinstance(arg, str) and arg.startswith("@"):
                    ref_name = arg[1:]  # Remove @
                    if ref_name in constants:
                        return constants[ref_name]
        return None


class CKKSMulHandler(BaseOperationHandler):
    """Handler for CKKS multiplication operations with automatic relinearization and rescaling."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle CKKS multiplication with relinearization AND rescaling."""
        from xdsl.dialects.builtin import DenseArrayBase

        # Get the other operand
        if operation.args and len(operation.args) > 0:
            other_operand = constants.get(f"arg_{hash(operation.args[0])}", operation.args[0])
        else:
            other_operand = current_value

        # Store original scaling factor
        # original_scale = type_builder.get_scaling_factor(current_value.type)

        # 1. Create multiplication operation (doubles scaling factor)
        result_type = type_builder.infer_result_type(
            "mul",
            current_value.type,
            other_operand.type if hasattr(other_operand, "type") else current_value.type,
        )

        mul_op = MulOp(operands=[current_value, other_operand], result_types=[result_type])
        block.add_op(mul_op)

        # 2. Add relinearization to reduce dimension back to 2
        relin_result_type = type_builder.create_ciphertext_type_with_dimension(
            2, preserve_from_type=mul_op.results[0].type
        )

        relin_op = RelinearizeOp(
            operands=[mul_op.results[0]],
            result_types=[relin_result_type],
            properties={
                "from_basis": DenseArrayBase.create_dense_int(i32, [0, 1, 2]),
                "to_basis": DenseArrayBase.create_dense_int(i32, [0, 1]),
            },
        )
        block.add_op(relin_op)

        return relin_op.results[0]
        # 3. Add rescale operation to reduce scaling factor back to original
        # rescaled_type = type_builder.create_rescaled_type(relin_op.results[0].type, original_scale)
        #
        # rescale_op = RescaleOp(
        #     operands=[relin_op.results[0]],
        #     result_types=[rescaled_type],
        #     properties={"to_ring": type_builder.get_next_modulus_ring(relin_op.results[0].type)}
        # )
        # block.add_op(rescale_op)
        #
        # return rescale_op.results[0]


class CKKSPlaintextHandler(BaseOperationHandler):
    """Handler for CKKS plaintext operations."""

    def __init__(self, op_class):
        self.op_class = op_class

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle CKKS plaintext operations (add_plain, mul_plain)."""

        print(f"🔧 Processing {operation.op_type} operation: {operation.result_var}")
        print(f"    Operation metadata: {operation.metadata}")

        # Get plaintext operand
        cleartext_operand = get_constant_operand(operation)

        # Extract bias into a new function argument
        ct_ty = current_value.type
        plaintext_type = LWEPlaintextType([ct_ty.application_data, ct_ty.plaintext_space])
        if operation.metadata["operation"] == "bias_addition":
            layer_name = operation.metadata.get("layer", "unknown_layer")
            func_op = get_parent_func(block)
            new_arg_index = len(func_op.args)
            new_arg_attrs = func_op.arg_attrs.data + (
                DictionaryAttr(
                    {
                        "orion.layer_name": StringAttr(layer_name),
                        "orion.layer_role": StringAttr("bias"),
                    }
                ),
            )
            plaintext = block.insert_arg(arg_type=plaintext_type, index=new_arg_index)
            func_op.properties["arg_attrs"] = ArrayAttr(new_arg_attrs)
            func_op.update_function_type()

            # FIXME: write plaintext constant data to disk for later loading
        else:
            plaintext = self._get_plaintext_operand(operation, constants)
            if not plaintext:
                raise ValueError(f"No plaintext operand found for {operation.op_type}")

        # Create the operation
        op_instance = self.op_class(
            operands=[current_value, plaintext], result_types=[current_value.type]
        )

        block.add_op(op_instance)
        mul_plain_result = op_instance.results[0]

        if operation.op_type == "mul_plain":
            # The mul_plain result has doubled scaling factor
            # Need to rescale back to original scaling factor

            # Get target scaling factor (usually the original ciphertext scaling)
            original_scale = type_builder.get_scaling_factor(current_value.type)

            # Create rescale operation
            rescale_result_type = type_builder.create_rescaled_type(
                mul_plain_result.type, original_scale
            )

            rescale_op = RescaleOp(operands=[mul_plain_result], result_types=[rescale_result_type])
            block.add_op(rescale_op)

            return rescale_op.results[0]

        print(f"✅ Created {self.op_class.name} operation")
        return op_instance.results[0]

    def _get_plaintext_operand(
        self, operation: FHEOperation, constants: Dict[str, SSAValue]
    ) -> SSAValue:
        """Get the plaintext operand."""

        # Check for special @ syntax in args
        if operation.args:
            for arg in operation.args:
                if isinstance(arg, str) and arg.startswith("@"):
                    # Reference to another operation's result
                    ref_name = arg[1:]  # Remove the @
                    if ref_name in constants:
                        print(f"    Found referenced result: {ref_name}")
                        return constants[ref_name]

        # Fallback to traditional patterns
        if operation.result_var:
            key = f"pt_{operation.result_var}_0"
            if key in constants:
                return constants[key]

        return None


class CKKSRotationHandler(BaseOperationHandler):
    """Handler for CKKS rotation operations."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle CKKS rotation operations."""
        # Extract rotation offset from operation
        offset = self._extract_rotation_offset(operation)

        # Result type is same as input for rotation
        result_type = current_value.type

        # Create rotation operation
        rotate_op = RotateOp(
            operands=[current_value],
            result_types=[result_type],
            properties={"offset": IntegerAttr(offset, IntegerType(32))},
        )

        block.add_op(rotate_op)
        return rotate_op.results[0]

    def _extract_rotation_offset(self, operation: FHEOperation) -> int:
        """Extract rotation offset from operation."""
        # Check kwargs first
        if "offset" in operation.kwargs:
            return operation.kwargs["offset"]

        # Check args
        if operation.args and len(operation.args) > 0:
            if isinstance(operation.args[0], int):
                return operation.args[0]

        # Default to 1 if no offset found
        return 1


class LWEEncodingHandler(BaseOperationHandler):
    """Handler for encoding operations with correct application data."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle encoding operations with matching application data."""
        print(f"🔧 Processing encode operation: {operation.result_var}")
        return current_value

        # if not operation.args:
        #     print("❌ No tensor argument found")
        #     return current_value

        # # Get the tensor and create constant
        # tensor_arg = operation.args[0]
        # print(f"    Encoding tensor with shape: {tensor_arg.shape}")

        # target_scale = None
        # if hasattr(operation, "metadata") and "target_scale" in operation.metadata:
        #     target_scale = operation.metadata["target_scale"]

        # encoded_plaintext = type_builder.create_slot_based_plaintext_encoding(
        #     block, tensor_arg, target_scale
        # )

        # slots = getattr(
        #     type_builder.scheme_params, "slots", type_builder.scheme_params.ring_degree // 2
        # )
        # print(f"✅ Created slot-based encoding (padded to {slots} slots)")

        # # Store result
        # if operation.result_var:
        #     constants[operation.result_var] = encoded_plaintext

        # return encoded_plaintext


class OperationRegistry:
    """
    Registry for FHE operation handlers.

    This allows for modular registration of different operation types
    and easy extension with new operations.
    """

    def __init__(self):
        self.handlers: Dict[str, OperationHandler] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default operation handlers."""
        # CKKS arithmetic operations
        self.handlers["add"] = CKKSArithmeticHandler(AddOp)
        self.handlers["sub"] = CKKSArithmeticHandler(SubOp)
        self.handlers["mul"] = CKKSMulHandler()

        # CKKS plaintext operations
        self.handlers["add_plain"] = CKKSPlaintextHandler(AddPlainOp)
        self.handlers["mul_plain"] = CKKSPlaintextHandler(MulPlainOp)

        # CKKS rotation operations
        self.handlers["rotate"] = CKKSRotationHandler()
        self.handlers["rot"] = CKKSRotationHandler()  # Alias

        # LWE operations
        self.handlers["encode"] = LWEEncodingHandler()

        # Linear transform operations (decomposed to rotations)
        self.handlers["linear_transform"] = LinearTransformHandler()

        self.handlers["quad"] = CKKSQuadHandler()

        self.handlers["orion.chebyshev"] = ChebyshevHandler()
        self.handlers["bootstrap"] = CKKSBootstrapHandler()

    def register_operation(self, op_type: str, handler: OperationHandler):
        """Register a custom operation handler."""
        self.handlers[op_type] = handler

    def translate_operation(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Translate an operation using the registered handler."""

        handler = self.handlers.get(operation.op_type)
        if handler is None:
            print(f"⚠️ No handler for operation type: {operation.op_type}")
            return current_value

        try:
            if hasattr(handler, "handle"):
                return handler.handle(operation, current_value, block, constants, type_builder)
            else:
                return handler(operation, current_value, block, constants, type_builder)
        except Exception as e:
            print(f"❌ Error translating operation {operation.op_type}: {e}")
            return current_value


# Fixed Orion Translator - Block-Based Linear Transform Handler


class LinearTransformHandler(BaseOperationHandler):
    """Handler for CKKS linear transform operations with block-based diagonal processing."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle CKKS linear transform with block-based diagonal processing."""
        print("🔧 LinearTransform handler: Processing Orion block-based linear transform")
        print(f"    Operation metadata: {operation.metadata}")

        # Extract Orion metadata
        orion_metadata = self._extract_orion_metadata(operation, type_builder)

        # Get the layer from operation args to extract diagonal blocks
        layer = None
        if operation.args and len(operation.args) > 0:
            layer = operation.args[0]

        # Create multiple linear transform operations - one per block
        if layer and hasattr(layer, "diagonals") and layer.diagonals:
            return self._handle_blocked_linear_transform(
                operation, current_value, block, constants, type_builder, layer, orion_metadata
            )
        else:
            # Fallback for single block or no diagonal data
            return self._handle_single_linear_transform(
                operation, current_value, block, constants, type_builder, orion_metadata
            )

    def _handle_blocked_linear_transform(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
        layer: Any,
        orion_metadata: Dict,
    ) -> SSAValue:
        """Handle linear transform with multiple blocks - create one operation per block."""

        diagonals = layer.diagonals
        # transform_ids = getattr(layer, 'transform_ids', {})

        print(f"    🔍 Processing {len(diagonals)} diagonal blocks")

        # Determine matrix block structure
        block_keys = list(diagonals.keys())
        if not block_keys:
            return self._handle_single_linear_transform(
                operation, current_value, block, constants, type_builder, orion_metadata
            )

        # Get dimensions
        max_row = max(key[0] for key in block_keys)
        max_col = max(key[1] for key in block_keys)
        num_block_rows = max_row + 1
        num_block_cols = max_col + 1

        print(f"    📊 Block matrix dimensions: {num_block_rows} x {num_block_cols}")

        # Create input tensor list for block columns
        input_tensors = self._create_input_tensor_list(
            current_value, num_block_cols, block, type_builder
        )

        # Process each block row
        block_row_results = []
        for row in range(num_block_rows):
            row_result = None

            # Process each block column in this row
            for col in range(num_block_cols):
                block_key = (row, col)

                if block_key in diagonals:
                    # Create linear transform for this block
                    block_result = self._create_block_linear_transform(
                        block_key,
                        diagonals[block_key],
                        input_tensors[col],
                        block,
                        type_builder,
                        orion_metadata,
                    )

                    # Accumulate results across columns
                    if row_result is None:
                        row_result = block_result
                    else:
                        row_result = self._add_ciphertexts(
                            row_result, block_result, block, type_builder
                        )

            if row_result is not None:
                block_row_results.append(row_result)

        # Combine all block row results
        if len(block_row_results) == 1:
            return block_row_results[0]
        else:
            return self._combine_block_results(block_row_results, block, type_builder)

    def _create_block_linear_transform(
        self,
        block_key: tuple,
        block_diagonals: Dict,
        input_tensor: SSAValue,
        block: Block,
        type_builder: Any,
        orion_metadata: Dict,
    ) -> SSAValue:
        """Create a single linear transform operation for one block."""

        row, col = block_key
        print(f"      🎯 Creating block linear transform for block ({row}, {col})")
        print(f"         Block contains {len(block_diagonals)} diagonals")

        # Stack all diagonals for this block (following Orion's pattern)
        diagonal_indices = []
        stacked_diagonal_data = []

        slots = orion_metadata.get("slots")

        for diag_idx in sorted(block_diagonals.keys()):
            diag_data = block_diagonals[diag_idx]

            # Convert to numpy array
            if hasattr(diag_data, "numpy"):
                diag_array = diag_data.detach().cpu().numpy()
            elif hasattr(diag_data, "__array__"):
                diag_array = np.array(diag_data)
            elif isinstance(diag_data, (list, tuple)):
                diag_array = np.array(diag_data)
            else:
                print(
                    f"         ⚠️  Skipping diagonal {diag_idx} with unknown type: {type(diag_data)}"
                )
                continue

            # Ensure correct shape and type
            diag_array = diag_array.astype(np.float32)
            if diag_array.size != slots:
                print(
                    f"         ⚠️  Diagonal {diag_idx} has size {diag_array.size}, expected {slots}"
                )
                if diag_array.size < slots:
                    # Pad with zeros
                    padded = np.zeros(slots, dtype=np.float32)
                    padded[: diag_array.size] = diag_array
                    diag_array = padded
                else:
                    # Truncate
                    diag_array = diag_array[:slots]

            diagonal_indices.append(diag_idx)
            stacked_diagonal_data.extend(diag_array.tolist())

        if not diagonal_indices:
            print(f"         ❌ No valid diagonals found for block ({row}, {col})")
            return input_tensor  # Return unchanged input

        print(f"         ✅ Stacked {len(diagonal_indices)} diagonals into single transform")

        # Create tensor arg for pre-packed plaintext diagonals
        # NOTE: the application_data field here is incorrect, but it doesn't
        # matter unless we want to use the debugging helper. See
        # https://github.com/google/heir/issues/2280
        plaintext_type = type_builder.get_default_plaintext_type()
        plaintext_tensor_shape = [len(diagonal_indices)]
        plaintext_tensor_type = TensorType(plaintext_type, plaintext_tensor_shape)

        func_op = get_parent_func(block)
        layer_name = orion_metadata.get("layer", "unknown_layer")
        new_arg_index = len(func_op.args)
        new_arg_attrs = func_op.arg_attrs.data + (
            DictionaryAttr(
                {
                    "orion.layer_name": StringAttr(layer_name),
                    "orion.layer_role": StringAttr("weights"),
                    "orion.block_row": IntegerAttr.from_int_and_width(row, 64),
                    "orion.block_col": IntegerAttr.from_int_and_width(col, 64),
                }
            ),
        )
        inserted_diagonals_block_arg = block.insert_arg(
            arg_type=plaintext_tensor_type, index=new_arg_index
        )
        func_op.properties["arg_attrs"] = ArrayAttr(new_arg_attrs)
        func_op.update_function_type()

        # FIXME: write stacked_diagonal_data to disk for later loading

        attributes = self._create_block_attributes(block_key, diagonal_indices, orion_metadata)
        result_type = type_builder.infer_plaintext_result_type(
            "mul_plain", input_tensor.type, inserted_diagonals_block_arg.type
        )
        linear_transform_op = LinearTransformOp(
            operands=[input_tensor, inserted_diagonals_block_arg],
            result_types=[result_type],
            attributes=attributes,
        )

        block.add_op(linear_transform_op)
        print(f"         ✅ Created block linear transform for ({row}, {col})")

        return linear_transform_op.results[0]

    def _create_input_tensor_list(
        self, current_value: SSAValue, num_block_cols: int, block: Block, type_builder: Any
    ) -> List[SSAValue]:
        """Create list of input tensors for block columns."""

        if num_block_cols == 1:
            return [current_value]

        # For multiple columns, we need to split/replicate the input
        # This is a simplified version - in practice, you might need more sophisticated splitting
        input_tensors = []

        for col in range(num_block_cols):
            # For now, use the same input for all columns
            # TODO: Implement proper input splitting based on block structure
            input_tensors.append(current_value)

        return input_tensors

    def _add_ciphertexts(
        self, left: SSAValue, right: SSAValue, block: Block, type_builder: Any
    ) -> SSAValue:
        """Add two ciphertexts."""
        from ..dialects.ckks import AddOp

        add_op = AddOp(operands=[left, right], result_types=[left.type])
        block.add_op(add_op)

        return add_op.results[0]

    def _combine_block_results(
        self, block_results: List[SSAValue], block: Block, type_builder: Any
    ) -> SSAValue:
        """Combine results from multiple block rows."""

        result = block_results[0]
        for i in range(1, len(block_results)):
            result = self._add_ciphertexts(result, block_results[i], block, type_builder)

        return result

    def _create_block_attributes(
        self, block_key: tuple, diagonal_indices: List[int], orion_metadata: Dict
    ) -> Dict:
        """Create attributes for a block linear transform."""
        from xdsl.dialects.builtin import IntegerAttr, IntegerType

        row, col = block_key
        attributes = {}

        # Block coordinates
        attributes["block_row"] = IntegerAttr(row, IntegerType(32))
        attributes["block_col"] = IntegerAttr(col, IntegerType(32))

        # Diagonal information
        attributes["diagonal_count"] = IntegerAttr(len(diagonal_indices), IntegerType(32))

        # Orion metadata
        if "slots" in orion_metadata:
            attributes["slots"] = IntegerAttr(orion_metadata["slots"], IntegerType(32))

        if "bsgs_ratio" in orion_metadata:
            attributes["bsgs_ratio"] = FloatAttr(orion_metadata["bsgs_ratio"], f64)

        if "orion_level" in orion_metadata:
            attributes["orion_level"] = IntegerAttr(orion_metadata["orion_level"], IntegerType(32))

        return attributes

    def _handle_single_linear_transform(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
        orion_metadata: Dict,
    ) -> SSAValue:
        """Fallback handler for single block or no diagonal data."""
        from ..dialects.ckks import LinearTransformOp

        print("    🔄 Fallback: Creating single linear transform operation")

        # Create simple linear transform operation
        attributes = self._create_attributes_from_metadata(orion_metadata, operation)

        result_type = current_value.type
        linear_transform_op = LinearTransformOp(
            operands=[current_value], result_types=[result_type], attributes=attributes
        )

        block.add_op(linear_transform_op)
        print("    ✅ Created single linear transform (fallback)")

        return linear_transform_op.results[0]

    def _extract_orion_metadata(self, operation: FHEOperation, type_builder: Any) -> Dict:
        """Extract Orion-specific metadata from the operation."""
        import math

        metadata = {}

        # Copy basic metadata
        if operation.metadata:
            metadata.update(operation.metadata)

        # Add default BSGS parameters if not present
        metadata.setdefault("bsgs_ratio", 2.0)
        slots = type_builder.scheme_params.ring_degree // 2
        metadata.setdefault("slots", slots)
        metadata.setdefault("embedding_method", "hybrid")
        metadata.setdefault("orion_level", operation.level or 1)

        # Calculate baby/giant step sizes
        # diagonal_count = metadata.get('diagonal_count', 128)
        bsgs_ratio = metadata.get("bsgs_ratio", 2.0)

        baby_step_size = int(math.sqrt(slots) / bsgs_ratio)
        giant_step_size = slots // baby_step_size

        metadata["baby_step_size"] = baby_step_size
        metadata["giant_step_size"] = giant_step_size

        return metadata

    def _create_attributes_from_metadata(
        self, orion_metadata: Dict, operation: FHEOperation
    ) -> Dict:
        """Create MLIR attributes from Orion metadata."""
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, FloatAttr, f64, StringAttr

        attributes = {}

        # Core parameters
        if "diagonal_count" in orion_metadata:
            attributes["diagonal_count"] = IntegerAttr(
                orion_metadata["diagonal_count"], IntegerType(32)
            )

        if "layer" in orion_metadata:
            attributes["layer_name"] = StringAttr(orion_metadata["layer"])

        if "bsgs_ratio" in orion_metadata:
            attributes["bsgs_ratio"] = FloatAttr(orion_metadata["bsgs_ratio"], f64)

        if "baby_step_size" in orion_metadata:
            attributes["baby_step_size"] = IntegerAttr(
                orion_metadata["baby_step_size"], IntegerType(32)
            )

        if "giant_step_size" in orion_metadata:
            attributes["giant_step_size"] = IntegerAttr(
                orion_metadata["giant_step_size"], IntegerType(32)
            )

        if "slots" in orion_metadata:
            attributes["slots"] = IntegerAttr(orion_metadata["slots"], IntegerType(32))

        if "matrix_shape" in orion_metadata:
            shape = orion_metadata["matrix_shape"]
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                attributes["matrix_rows"] = IntegerAttr(shape[0], IntegerType(32))
                attributes["matrix_cols"] = IntegerAttr(shape[1], IntegerType(32))

        if "orion_level" in orion_metadata:
            attributes["orion_level"] = IntegerAttr(orion_metadata["orion_level"], IntegerType(32))

        return attributes


class CKKSQuadHandler(BaseOperationHandler):
    """Handler for CKKS quadratic activation operations."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle quadratic activation: x * x."""
        print(f"🔢 Processing quadratic activation: {operation.result_var}")

        # original_scale = type_builder.get_scaling_factor(current_value.type)
        result_type = type_builder.infer_result_type_with_relinearization(
            "mul", current_value.type, current_value.type
        )

        # Create self-multiplication operation
        quad_op = MulOp(
            operands=[current_value, current_value], result_types=[result_type]  # x * x
        )

        block.add_op(quad_op)
        print("✅ Created ckks.mul operation (x * x)")

        # Add relinearization to reduce dimension back to 2
        relin_result_type = type_builder.create_relinearized_ciphertext_type(
            quad_op.results[0].type
        )

        relin_op = RelinearizeOp(
            operands=[quad_op.results[0]],
            result_types=[relin_result_type],
            properties={
                "from_basis": DenseArrayBase.create_dense_int(i32, [0, 1, 2]),
                "to_basis": DenseArrayBase.create_dense_int(i32, [0, 1]),
            },
        )
        block.add_op(relin_op)

        print("✅ Created ckks.mul + ckks.relinearize operations (x * x)")
        return relin_op.results[0]
        # rescaled_type = type_builder.create_rescaled_type(relin_op.results[0].type, original_scale)
        #
        # rescale_op = RescaleOp(
        #     operands=[relin_op.results[0]],
        #     result_types=[rescaled_type],
        #     properties={"to_ring": type_builder.get_next_modulus_ring(relin_op.results[0].type)}
        # )
        # block.add_op(rescale_op)

        # return rescale_op.results[0]


class ChebyshevHandler(BaseOperationHandler):
    """Handler for Chebyshev polynomial evaluation operations."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle Chebyshev polynomial evaluation."""
        from ..dialects.orion import ChebyshevOp
        from ..dialects.ckks import BootstrapOp
        from xdsl.dialects.builtin import ArrayAttr, FloatAttr, f64

        # Get coefficients from operation
        coeffs = operation.kwargs.get("coefficients", [])
        domain_start = operation.kwargs.get("domain_start", -1.0)
        domain_end = operation.kwargs.get("domain_end", 1.0)

        print(f"🔧 Creating Chebyshev operation with {len(coeffs)} coefficients")

        if not coeffs:
            print("⚠️ No coefficients provided for Chebyshev operation")
            return current_value

        # Create coefficient attributes
        coeff_attrs = [FloatAttr(float(c), f64) for c in coeffs]
        coeff_array = ArrayAttr(coeff_attrs)

        # Create result type
        bootstrap_result_type = type_builder.get_default_ciphertext_type()

        # FIXME: use Orion's bootstrap placement
        # Create bootstrap operation
        bootstrap_op = BootstrapOp(operands=[current_value], result_types=[bootstrap_result_type])

        block.add_op(bootstrap_op)
        bootstrapped_value = bootstrap_op.results[0]
        result_type = type_builder.get_default_ciphertext_type()
        # Create Chebyshev operation
        cheby_op = ChebyshevOp(
            operands=[bootstrapped_value],
            result_types=[result_type],
            properties={
                "coefficients": coeff_array,
                "domain_start": FloatAttr(domain_start, f64),
                "domain_end": FloatAttr(domain_end, f64),
            },
        )

        block.add_op(cheby_op)
        print("✅ Created ckks.chebyshev operation")

        # Store result
        if operation.result_var:
            constants[operation.result_var] = cheby_op.results[0]

        return cheby_op.results[0]


class CKKSBootstrapHandler(BaseOperationHandler):
    """Handler for CKKS Bootstrap (noise refresh) operations."""

    def handle(
        self,
        operation: FHEOperation,
        current_value: SSAValue,
        block: Block,
        constants: Dict[str, SSAValue],
        type_builder: Any,
    ) -> SSAValue:
        """Handle bootstrap (refresh) operation."""
        from ..dialects.ckks import BootstrapOp

        print("🔧 Creating Bootstrap operation")

        # Create result type (bootstrap typically resets to fresh ciphertext)
        result_type = type_builder.get_default_ciphertext_type()

        # Create bootstrap operation
        bootstrap_op = BootstrapOp(operands=[current_value], result_types=[result_type])

        block.add_op(bootstrap_op)
        print("✅ Created ckks.bootstrap operation")

        # Store result
        if operation.result_var:
            constants[operation.result_var] = bootstrap_op.results[0]

        return bootstrap_op.results[0]
