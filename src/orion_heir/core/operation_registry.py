"""
Operation registry for modular FHE operation translation.

This module provides a registry system for different FHE operations,
allowing for easy extension and customization of translation behavior.
"""

from typing import Dict, Callable, Any, Protocol, List, Optional
from abc import ABC, abstractmethod

from xdsl.ir import SSAValue, Block
from xdsl.dialects.builtin import IntegerAttr, IntegerType

from .translator import FHEOperation

import torch


class OperationHandler(Protocol):
    """Protocol for operation handlers."""
    
    def __call__(self, 
                operation: FHEOperation,
                current_value: SSAValue,
                block: Block,
                constants: Dict[str, SSAValue],
                type_builder: Any) -> SSAValue:
        """Handle the translation of an FHE operation."""
        ...


class BaseOperationHandler(ABC):
    """Base class for operation handlers."""
    
    @abstractmethod
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle the translation of an FHE operation."""
        pass


class CKKSArithmeticHandler(BaseOperationHandler):
    """Handler for CKKS arithmetic operations (add, sub, mul)."""
    
    def __init__(self, op_class):
        self.op_class = op_class
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle CKKS arithmetic operations."""
        from ..dialects.ckks import AddOp, SubOp, MulOp
        
        # Determine operands
        if operation.op_type in ['add', 'sub', 'mul']:
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
                operands=[current_value, second_operand],
                result_types=[result_type]
            )
            
            block.add_op(op_instance)
            return op_instance.results[0]
        
        return current_value
    
    def _get_second_operand(self, operation: FHEOperation, 
                           constants: Dict[str, SSAValue]) -> SSAValue:
        """Get the second operand for binary operations."""
        # Look for constants or other operands
        if operation.result_var and f"pt_{operation.result_var}_0" in constants:
            return constants[f"pt_{operation.result_var}_0"]
        
        # Check for @ references in args
        if operation.args:
            for arg in operation.args:
                if isinstance(arg, str) and arg.startswith('@'):
                    ref_name = arg[1:]  # Remove @
                    if ref_name in constants:
                        return constants[ref_name]
        return None



class CKKSPlaintextHandler(BaseOperationHandler):
    """Handler for CKKS plaintext operations."""
    
    def __init__(self, op_class):
        self.op_class = op_class
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle CKKS plaintext operations (add_plain, mul_plain)."""
        
        print(f"🔧 Processing {operation.op_type} operation: {operation.result_var}")
        
        # Get plaintext operand
        plaintext = self._get_plaintext_operand(operation, constants)
        if plaintext is None:
            print(f"❌ No plaintext operand found for {operation.op_type}")
            return current_value
        
        print(f"✅ Found plaintext operand")
        
        # Create the operation
        op_instance = self.op_class(
            operands=[current_value, plaintext],
            result_types=[current_value.type]
        )
        
        block.add_op(op_instance)
        print(f"✅ Created {self.op_class.name} operation")
        return op_instance.results[0]
    
    def _get_plaintext_operand(self, operation: FHEOperation,
                              constants: Dict[str, SSAValue]) -> SSAValue:
        """Get the plaintext operand."""
        
        # Check for special @ syntax in args
        if operation.args:
            for arg in operation.args:
                if isinstance(arg, str) and arg.startswith('@'):
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
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle CKKS rotation operations."""
        from ..dialects.ckks import RotateOp
        
        # Extract rotation offset from operation
        offset = self._extract_rotation_offset(operation)
        
        # Result type is same as input for rotation
        result_type = current_value.type
        
        # Create rotation operation
        rotate_op = RotateOp(
            operands=[current_value],
            result_types=[result_type],
            properties={"offset": IntegerAttr(offset, IntegerType(32))}
        )
        
        block.add_op(rotate_op)
        return rotate_op.results[0]
    
    def _extract_rotation_offset(self, operation: FHEOperation) -> int:
        """Extract rotation offset from operation."""
        # Check kwargs first
        if 'offset' in operation.kwargs:
            return operation.kwargs['offset']
        
        # Check args
        if operation.args and len(operation.args) > 0:
            if isinstance(operation.args[0], int):
                return operation.args[0]
        
        # Default to 1 if no offset found
        return 1


class LWEEncodingHandler(BaseOperationHandler):
    """Simple handler for encoding operations."""
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle encoding operations."""
        from ..dialects.lwe import RLWEEncodeOp
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32
        from xdsl.dialects.arith import ConstantOp
        import torch
        import numpy as np
        
        print(f"🔧 Processing encode operation: {operation.result_var}")
        
        if not operation.args:
            print(f"❌ No tensor argument found")
            return current_value
        
        # Get the tensor and create constant
        tensor_arg = operation.args[0]
        print(f"    Encoding tensor with shape: {tensor_arg.shape}")
        
        # Convert tensor to constant
        if isinstance(tensor_arg, torch.Tensor):
            tensor_np = tensor_arg.detach().cpu().numpy().astype(np.float32)
        else:
            tensor_np = np.array(tensor_arg, dtype=np.float32)
        
        tensor_shape = list(tensor_np.shape)
        tensor_type = TensorType(f32, tensor_shape)
        flat_data = [float(x) for x in tensor_np.flatten()]
        
        dense_attr = DenseIntOrFPElementsAttr.create_dense_float(tensor_type, flat_data)
        const_op = ConstantOp(dense_attr, tensor_type)
        block.add_op(const_op)
        
        # Encode to plaintext
        plaintext_type = type_builder.get_default_plaintext_type()
        encode_op = RLWEEncodeOp(
            operands=[const_op.results[0]],
            result_types=[plaintext_type]
        )
        block.add_op(encode_op)
        
        print(f"✅ Created encoding operations")
        
        # Store result for future reference
        if operation.result_var:
            constants[operation.result_var] = encode_op.results[0]
        
        return encode_op.results[0]




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
        from ..dialects.ckks import AddOp, SubOp, MulOp, AddPlainOp, MulPlainOp
        
        # CKKS arithmetic operations
        self.handlers['add'] = CKKSArithmeticHandler(AddOp)
        self.handlers['sub'] = CKKSArithmeticHandler(SubOp)
        self.handlers['mul'] = CKKSArithmeticHandler(MulOp)
        
        # CKKS plaintext operations
        self.handlers['add_plain'] = CKKSPlaintextHandler(AddPlainOp)
        self.handlers['mul_plain'] = CKKSPlaintextHandler(MulPlainOp)
        
        # CKKS rotation operations
        self.handlers['rotate'] = CKKSRotationHandler()
        self.handlers['rot'] = CKKSRotationHandler()  # Alias
        
        # LWE operations
        self.handlers['encode'] = LWEEncodingHandler()
        
        # Linear transform operations (decomposed to rotations)
        self.handlers['linear_transform'] = CKKSLinearTransformHandler() 

        self.handlers['quad'] = CKKSQuadHandler()

    def register_operation(self, op_type: str, handler: OperationHandler):
        """Register a custom operation handler."""
        self.handlers[op_type] = handler
    
    def translate_operation(self, 
                           operation: FHEOperation,
                           current_value: SSAValue,
                           block: Block,
                           constants: Dict[str, SSAValue],
                           type_builder: Any) -> SSAValue:
        """Translate an operation using the registered handler."""
        
        handler = self.handlers.get(operation.op_type)
        if handler is None:
            print(f"⚠️ No handler for operation type: {operation.op_type}")
            return current_value
        
        try:
            if hasattr(handler, 'handle'):
                return handler.handle(operation, current_value, block, constants, type_builder)
            else:
                return handler(operation, current_value, block, constants, type_builder)
        except Exception as e:
            print(f"❌ Error translating operation {operation.op_type}: {e}")
            return current_value


class CKKSLinearTransformHandler(BaseOperationHandler):
    """Handler for CKKS linear transform operations with plaintext weights."""
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle CKKS linear transform operations with both ciphertext and plaintext inputs."""
        from ..dialects.ckks import LinearTransformOp
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, ArrayAttr, FloatAttr, f32, StringAttr
        
        print(f"🔧 LinearTransform handler: Processing Orion linear transform")
        print(f"    Operation metadata: {operation.metadata}")
        
        # Extract Orion metadata
        orion_metadata = self._extract_orion_metadata(operation)
        
        # Step 1: Create the plaintext weights from Orion diagonal data
        weights_plaintext = self._create_diagonal_plaintexts(operation, orion_metadata, block, type_builder)
        
        # Step 2: Create attributes from Orion metadata
        attributes = self._create_attributes_from_metadata(orion_metadata, operation)
        
        # Step 3: Create the linear transform operation with both inputs
        result_type = current_value.type
        
        try:
            linear_transform_op = LinearTransformOp(
                operands=[current_value, weights_plaintext],  # Ciphertext + plaintext weights
                result_types=[result_type],
                attributes=attributes
            )
            
            block.add_op(linear_transform_op)
            print(f"✅ Created ckks.linear_transform operation with ciphertext and plaintext weights")
            print(f"    📋 Attributes: {list(attributes.keys())}")
            
            return linear_transform_op.results[0]
            
        except Exception as e:
            print(f"❌ Error creating LinearTransformOp: {e}")
            # Fallback to single input for now
            linear_transform_op = LinearTransformOp(
                operands=[current_value],
                result_types=[result_type]
            )
            block.add_op(linear_transform_op)
            print(f"⚠️  Created LinearTransformOp with single input (fallback)")
            return linear_transform_op.results[0]
    
    def _create_diagonal_plaintexts(self, operation: FHEOperation, orion_metadata: Dict, 
                                   block: Block, type_builder: Any) -> SSAValue:
        """
        Create plaintext weights from Orion diagonal data.
        
        In Orion, the linear transform uses precomputed diagonal plaintexts.
        We need to extract these and encode them as LWE plaintexts.
        """
        from ..dialects.lwe import RLWEEncodeOp
        from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32
        from xdsl.dialects.arith import ConstantOp
        import torch
        import numpy as np
        
        print(f"    🔧 Creating diagonal plaintexts...")
        
        # Try to get diagonal data from operation args
        diagonal_data = None
        if operation.args and len(operation.args) > 0:
            # If diagonal data was passed as an argument
            diagonal_data = operation.args[0]
            print(f"    📊 Found diagonal data in operation args: {type(diagonal_data)}")
        
        # If no diagonal data in args, create placeholder
        if diagonal_data is None:
            diagonal_count = orion_metadata.get('diagonal_count', 128)
            slots = orion_metadata.get('slots', 4096)
            
            # Create placeholder diagonal data (zeros)
            # In a real implementation, this would come from Orion's diagonals
            diagonal_data = torch.zeros(diagonal_count, slots, dtype=torch.float32)
            print(f"    ⚠️  Created placeholder diagonal data: {diagonal_data.shape}")
        
        # Convert to numpy if it's a tensor
        if isinstance(diagonal_data, torch.Tensor):
            diagonal_np = diagonal_data.detach().cpu().numpy().astype(np.float32)
        else:
            diagonal_np = np.array(diagonal_data, dtype=np.float32)
        
        # Create tensor type and constant
        tensor_shape = list(diagonal_np.shape)
        tensor_type = TensorType(f32, tensor_shape)
        flat_data = [float(x) for x in diagonal_np.flatten()]
        
        # Create dense attribute and constant operation
        dense_attr = DenseIntOrFPElementsAttr.create_dense_float(tensor_type, flat_data)
        const_op = ConstantOp(dense_attr, tensor_type)
        block.add_op(const_op)
        
        # Encode to LWE plaintext
        plaintext_type = type_builder.get_default_plaintext_type()
        encode_op = RLWEEncodeOp(
            operands=[const_op.results[0]],
            result_types=[plaintext_type]
        )
        block.add_op(encode_op)
        
        print(f"    ✅ Created diagonal plaintexts: {tensor_shape}")
        return encode_op.results[0]
    
    def _create_attributes_from_metadata(self, orion_metadata: Dict, operation: FHEOperation) -> Dict:
        """Create MLIR attributes from Orion metadata."""
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, ArrayAttr, FloatAttr, f32, StringAttr
        
        attributes = {}
        
        # Core parameters
        if 'diagonal_count' in orion_metadata:
            attributes['diagonal_count'] = IntegerAttr(orion_metadata['diagonal_count'], IntegerType(32))
        
        if 'layer' in orion_metadata:
            attributes['layer_name'] = StringAttr(orion_metadata['layer'])
        
        if 'bsgs_ratio' in orion_metadata:
            attributes['bsgs_ratio'] = FloatAttr(orion_metadata['bsgs_ratio'], f32)
        
        if 'baby_step_size' in orion_metadata:
            attributes['baby_step_size'] = IntegerAttr(orion_metadata['baby_step_size'], IntegerType(32))
        
        if 'giant_step_size' in orion_metadata:
            attributes['giant_step_size'] = IntegerAttr(orion_metadata['giant_step_size'], IntegerType(32))
        
        if 'slots' in orion_metadata:
            attributes['slots'] = IntegerAttr(orion_metadata['slots'], IntegerType(32))
        
        if 'matrix_shape' in orion_metadata:
            shape = orion_metadata['matrix_shape']
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                attributes['matrix_rows'] = IntegerAttr(shape[0], IntegerType(32))
                attributes['matrix_cols'] = IntegerAttr(shape[1], IntegerType(32))
        
        if operation.level:
            attributes['orion_level'] = IntegerAttr(operation.level, IntegerType(32))
        
        return attributes
    
    def _extract_orion_metadata(self, operation: FHEOperation) -> Dict:
        """Extract Orion-specific metadata from the operation."""
        import math
        
        metadata = {}
        
        # Copy basic metadata
        if operation.metadata:
            metadata.update(operation.metadata)
        
        # Add default BSGS parameters if not present
        metadata.setdefault('bsgs_ratio', 2.0)
        metadata.setdefault('slots', 4096)
        metadata.setdefault('embedding_method', 'hybrid')
        
        # Calculate baby/giant step sizes
        diagonal_count = metadata.get('diagonal_count', 128)
        slots = metadata.get('slots', 4096)
        bsgs_ratio = metadata.get('bsgs_ratio', 2.0)
        
        baby_step_size = int(math.sqrt(slots) / bsgs_ratio)
        giant_step_size = slots // baby_step_size
        
        metadata['baby_step_size'] = baby_step_size
        metadata['giant_step_size'] = giant_step_size
        
        return metadata

def extract_orion_diagonals(layer: Any) -> Optional[torch.Tensor]:
    """
    Extract diagonal data from Orion layer with comprehensive checking.
    """
    import torch
    import numpy as np
    
    if not hasattr(layer, 'diagonals') or not layer.diagonals:
        print(f"      ❌ No diagonals attribute or empty diagonals")
        return None
    
    print(f"      🔍 Orion diagonals structure:")
    print(f"         Blocks: {list(layer.diagonals.keys())}")
    
    # Get the first block
    first_block_key = next(iter(layer.diagonals.keys()))
    first_block = layer.diagonals[first_block_key]
    
    if not first_block:
        print(f"         ❌ First block is empty")
        return None
    
    print(f"         First block {first_block_key}: {len(first_block)} diagonals")
    
    # Extract diagonal data
    diagonal_list = []
    diagonal_indices = sorted(first_block.keys())
    
    for diag_idx in diagonal_indices:  # Check first 5 diagonals
        diag_data = first_block[diag_idx]
        
        print(f"         Diagonal {diag_idx}: type={type(diag_data)}")
        
        # Handle different data types
        if diag_data is None:
            print(f"         ❌ Diagonal {diag_idx} is None")
            continue
        elif isinstance(diag_data, (list, tuple)) and len(diag_data) == 0:
            print(f"         ❌ Diagonal {diag_idx} is empty list/tuple")
            continue
        elif hasattr(diag_data, 'numel') and diag_data.numel() == 0:
            print(f"         ❌ Diagonal {diag_idx} is empty tensor")
            continue
        
        # Convert to numpy array
        if hasattr(diag_data, 'numpy'):
            diag_array = diag_data.detach().cpu().numpy()
        elif hasattr(diag_data, '__array__'):
            diag_array = np.array(diag_data)
        elif isinstance(diag_data, (list, tuple)):
            diag_array = np.array(diag_data)
        else:
            print(f"         ❌ Unknown diagonal data type: {type(diag_data)}")
            continue
        
        diag_array = diag_array.astype(np.float32)
        
        # Check if it's meaningful data
        if diag_array.size == 0:
            print(f"         ❌ Diagonal {diag_idx} is empty array")
            continue
        elif np.allclose(diag_array, 0.0):
            print(f"         ⚠️  Diagonal {diag_idx} is all zeros (might be valid)")
        else:
            print(f"         ✅ Diagonal {diag_idx}: shape={diag_array.shape}, range=[{diag_array.min():.6f}, {diag_array.max():.6f}]")
        
        diagonal_list.append(diag_array)
    
    if not diagonal_list:
        print(f"      ❌ No valid diagonal data found")
        return None
    
    # Stack all diagonals
    try:
        diagonal_array = np.stack(diagonal_list, axis=0)
        diagonal_tensor = torch.from_numpy(diagonal_array)
        
        print(f"      ✅ Extracted diagonal tensor: {diagonal_tensor.shape}")
        print(f"         Data range: [{diagonal_tensor.min():.6f}, {diagonal_tensor.max():.6f}]")
        
        return diagonal_tensor
    except Exception as e:
        print(f"      ❌ Error stacking diagonals: {e}")
        return None


class CKKSQuadHandler(BaseOperationHandler):
    """Handler for CKKS quadratic activation operations."""
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle quadratic activation: x * x."""
        from ..dialects.ckks import MulOp
        
        print(f"🔢 Processing quadratic activation: {operation.result_var}")
        
        # Create self-multiplication operation
        quad_op = MulOp(
            operands=[current_value, current_value],  # x * x
            result_types=[current_value.type]
        )
        
        block.add_op(quad_op)
        print(f"✅ Created ckks.mul operation (x * x)")
        
        return quad_op.results[0]
