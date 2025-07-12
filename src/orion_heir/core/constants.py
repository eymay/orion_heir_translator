"""
Constants manager for handling constant creation in HEIR translation.

This module manages the creation of constant values and plaintext encodings
needed during the translation process.
"""

from typing import Dict, List, Any
import torch
import numpy as np

from xdsl.ir import SSAValue, Block
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f64
from xdsl.dialects.arith import ConstantOp

from .types import FHEOperation
from .type_builder import TypeBuilder


class ConstantManager:
    """
    Manager for creating and tracking constants during translation.
    
    This class handles the creation of plaintext constants and their
    encoding into the appropriate HEIR types.
    """
    
    def __init__(self, type_builder: TypeBuilder):
        self.type_builder = type_builder
        self.created_constants: Dict[str, SSAValue] = {}
    
    def create_constants(self, block: Block, operations: List[FHEOperation]) -> Dict[str, SSAValue]:
        """
        Create all constants needed for the given operations.
        
        Args:
            block: MLIR block to add constants to
            operations: List of operations that may need constants
            
        Returns:
            Dictionary mapping constant names to their SSA values
        """
        constants = {}
        
        for i, operation in enumerate(operations):
            # Create constants for tensor arguments
            for j, arg in enumerate(operation.args or []):
                if isinstance(arg, torch.Tensor):
                    const_name = f"pt_{operation.result_var}_{j}" if operation.result_var else f"constant_{i}_{j}"
                    if const_name not in constants:
                        constants[const_name] = self._create_tensor_constant(block, arg)
            
            # Create constants for scalar arguments
            for j, arg in enumerate(operation.args or []):
                if isinstance(arg, (int, float)) and not isinstance(arg, bool):
                    const_name = f"scalar_{operation.result_var}_{j}" if operation.result_var else f"scalar_{i}_{j}"
                    if const_name not in constants:
                        constants[const_name] = self._create_scalar_constant(block, arg)
        
        return constants
    
    def _create_tensor_constant(self, block: Block, tensor: torch.Tensor) -> SSAValue:
        """Create a constant operation for a PyTorch tensor."""
        # Convert tensor to numpy and ensure it's float32
        tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
        
        # Get the tensor shape
        tensor_shape = list(tensor_np.shape)
        tensor_type = TensorType(f64, tensor_shape)
        
        # Flatten the data and convert to Python floats
        flat_data = tensor_np.flatten()
        float_data = [float(x) for x in flat_data]
        
        # Create dense attribute
        try:
            dense_attr = DenseIntOrFPElementsAttr.create_dense_float(
                tensor_type, float_data
            )
        except Exception as e:
            print(f"❌ Error creating dense attribute: {e}")
            print(f"   Tensor shape: {tensor_shape}")
            print(f"   Data length: {len(float_data)}")
            print(f"   Sample data: {float_data[:5] if len(float_data) > 5 else float_data}")
            raise
        
        # Create constant operation
        const_op = ConstantOp(dense_attr, tensor_type)
        block.add_op(const_op)
        
        return const_op.results[0]
    
    def _create_scalar_constant(self, block: Block, value: float) -> SSAValue:
        """Create a constant operation for a scalar value."""
        # Create a 1D tensor with single element
        import numpy as np
        
        padded_data = np.zeros(self.slots, dtype=np.float32)
        padded_data[0] = float(value)
        
        return self.type_builder.create_padded_tensor_constant(block, padded_data, self.slots)
    
    
    def create_plaintext_encoding(self, block: Block, constant_value: SSAValue) -> SSAValue:
        """
        Create a plaintext encoding operation for a constant.
        
        This encodes a constant tensor into the LWE plaintext space.
        """
        return self.type_builder.create_slot_based_plaintext_encoding(block, tensor_value)
    
    def get_or_create_constant(self, block: Block, key: str, value: Any) -> SSAValue:
        """Get an existing constant or create a new one."""
        if key in self.created_constants:
            return self.created_constants[key]
        
        if isinstance(value, torch.Tensor):
            constant = self._create_tensor_constant(block, value)
        elif isinstance(value, (int, float)):
            constant = self._create_scalar_constant(block, value)
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")
        
        self.created_constants[key] = constant
        return constant


def extract_constants_from_operations(operations: List[FHEOperation]) -> Dict[str, Any]:
    """
    Extract all constant values from a list of operations.
    
    Args:
        operations: List of FHE operations
        
    Returns:
        Dictionary mapping constant identifiers to their values
    """
    constants = {}
    
    for i, operation in enumerate(operations):
        # Extract tensor constants
        for j, arg in enumerate(operation.args or []):
            if isinstance(arg, torch.Tensor):
                key = f"tensor_{i}_{j}"
                constants[key] = arg
        
        # Extract scalar constants
        for j, arg in enumerate(operation.args or []):
            if isinstance(arg, (int, float)) and not isinstance(arg, bool):
                key = f"scalar_{i}_{j}"
                constants[key] = arg
    
    return constants
