"""
Operation registry for modular FHE operation translation.

This module provides a registry system for different FHE operations,
allowing for easy extension and customization of translation behavior.
"""

from typing import Dict, Callable, Any, Protocol
from abc import ABC, abstractmethod

from xdsl.ir import SSAValue, Block
from xdsl.dialects.builtin import IntegerAttr, IntegerType

from .translator import FHEOperation


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
        
        # Could also look in operation.args for other ciphertexts
        # For now, return None if no second operand found
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
        from ..dialects.ckks import AddPlainOp, MulPlainOp
        
        # Get plaintext operand
        plaintext = self._get_plaintext_operand(operation, constants)
        if plaintext is None:
            return current_value
        
        # Determine result type
        result_type = type_builder.infer_plaintext_result_type(
            operation.op_type, current_value.type, plaintext.type
        )
        
        # Create the operation
        op_instance = self.op_class(
            operands=[current_value, plaintext],
            result_types=[result_type]
        )
        
        block.add_op(op_instance)
        return op_instance.results[0]
    
    def _get_plaintext_operand(self, operation: FHEOperation,
                              constants: Dict[str, SSAValue]) -> SSAValue:
        """Get the plaintext operand."""
        if operation.result_var and f"pt_{operation.result_var}_0" in constants:
            return constants[f"pt_{operation.result_var}_0"]
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
    """Handler for LWE encoding operations."""
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle LWE encoding operations."""
        from ..dialects.lwe import RLWEEncodeOp
        
        # Get the plaintext to encode
        plaintext_key = f"pt_{operation.result_var}_0" if operation.result_var else "constant_0"
        plaintext_type = type_builder.get_default_plaintext_type()

        if plaintext_key in constants:
            plaintext = constants[plaintext_key]
            
            # Create encoding operation
            encode_op = RLWEEncodeOp(
                operands=[plaintext],
                result_types=[plaintext_type]
            )
            
            block.add_op(encode_op)
            return encode_op.results[0]
        
        return current_value


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
    """Handler for CKKS linear transform operations."""
    
    def handle(self, 
              operation: FHEOperation,
              current_value: SSAValue,
              block: Block,
              constants: Dict[str, SSAValue],
              type_builder: Any) -> SSAValue:
        """Handle CKKS linear transform operations."""
        from ..dialects.ckks import LinearTransformOp
        
        print(f"🔧 LinearTransform handler: Processing Orion linear transform")
        print(f"    Operation metadata: {operation.metadata}")
        
        # Determine result type (same as input for now)
        result_type = current_value.type
        
        # Create the linear transform operation
        # This represents Orion's diagonal-based matrix multiplication
        linear_transform_op = LinearTransformOp(
            operands=[current_value],
            result_types=[result_type]
        )
        
        block.add_op(linear_transform_op)
        print(f"✅ Created ckks.linear_transform operation")
        return linear_transform_op.results[0]

