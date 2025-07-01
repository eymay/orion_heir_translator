"""
Generic translator infrastructure for converting FHE operations to HEIR MLIR.

This module provides the core translation framework that can be used by
different frontends (Orion, OpenFHE, SEAL, etc.).
"""

from typing import Dict, List, Any

from xdsl.ir import SSAValue, Block, Region
from xdsl.dialects.builtin import ModuleOp, FunctionType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.context import Context

from .types import FHEOperation, SchemeParameters, FrontendInterface
from .operation_registry import OperationRegistry
from .type_builder import TypeBuilder
from .constants import ConstantManager


class GenericTranslator:
    """
    Generic translator for converting FHE operations to HEIR MLIR.
    
    This class provides the core translation logic that can be used by
    any frontend. It uses an operation registry to handle different
    operation types in a modular way.
    """
    
    def __init__(self):
        self.context = Context()
        self.operation_registry = OperationRegistry()
        self._register_dialects()
    
    def _register_dialects(self):
        """Register all HEIR dialects with the context."""
        # Import dialects here to avoid circular imports
        from ..dialects.ckks import CKKS
        from ..dialects.lwe import LWE
        from ..dialects.polynomial import Polynomial
        from ..dialects.mod_arith import ModArith
        from ..dialects.rns import RNS
        from ..dialects.mgmt import MGMT
        
        dialects = [CKKS, LWE, Polynomial, ModArith, RNS, MGMT]
        for dialect in dialects:
            self.context.load_dialect(dialect)
    
    def translate(self, 
                  operations: List[FHEOperation], 
                  scheme_params: SchemeParameters,
                  function_name: str = "fhe_computation") -> ModuleOp:
        """
        Translate a list of FHE operations to HEIR MLIR.
        
        Args:
            operations: List of FHE operations to translate
            scheme_params: FHE scheme parameters
            function_name: Name for the generated function
            
        Returns:
            Complete MLIR module containing the translated operations
        """
        print(f"🔄 Translating {len(operations)} operations to HEIR...")
        
        # Build type system based on scheme parameters
        type_builder = TypeBuilder(scheme_params)
        
        # Create module with scheme parameters
        module = self._create_module(scheme_params, type_builder)
        
        # Create function containing the operations
        func = self._create_function(operations, type_builder, function_name)
        module.body.block.add_op(func)
        
        print("✅ Translation completed")
        return module
    
    def _create_module(self, scheme_params: SchemeParameters, 
                      type_builder: TypeBuilder) -> ModuleOp:
        """Create the top-level MLIR module with scheme attributes."""
        # Create module attributes based on scheme parameters
        attributes = type_builder.create_module_attributes()
        
        return ModuleOp([], attributes)
    
    def _create_function(self, 
                        operations: List[FHEOperation],
                        type_builder: TypeBuilder,
                        function_name: str) -> FuncOp:
        """Create a function containing the translated operations."""
        
        # Determine input and output types
        input_type = type_builder.get_default_ciphertext_type()
        
        # Create function with placeholder return type (will be corrected)
        func_type = FunctionType.from_lists([input_type], [input_type])
        func = FuncOp(
            name=function_name,
            function_type=func_type,
            region=Region.DEFAULT,
            visibility=None
        )
        
        entry_block = func.body.blocks.first
        
        # Create constants for the operations
        constant_manager = ConstantManager(type_builder)
        constants = constant_manager.create_constants(entry_block, operations)
        
        # Translate operations sequentially
        current_value = entry_block.args[0]  # Function input
        
        for i, operation in enumerate(operations):
            print(f"  Translating operation {i+1}/{len(operations)}: {operation.op_type}")
            current_value = self.operation_registry.translate_operation(
                operation, current_value, entry_block, constants, type_builder
            )
        
        # Update function type with actual return type
        actual_return_type = current_value.type
        corrected_func_type = FunctionType.from_lists([input_type], [actual_return_type])
        func.function_type = corrected_func_type
        
        # Add return statement
        entry_block.add_op(ReturnOp(current_value))
        
        return func


class TranslatorBuilder:
    """Builder class for creating configured translators."""
    
    def __init__(self):
        self.translator = GenericTranslator()
    
    def with_custom_operations(self, operations: Dict[str, Any]) -> 'TranslatorBuilder':
        """Add custom operation handlers."""
        for op_name, handler in operations.items():
            self.translator.operation_registry.register_operation(op_name, handler)
        return self
    
    def with_frontend(self, frontend: FrontendInterface) -> 'TranslatorBuilder':
        """Configure with a specific frontend."""
        self.frontend = frontend
        return self
    
    def build(self) -> GenericTranslator:
        """Build the configured translator."""
        return self.translator


def create_translator() -> GenericTranslator:
    """Create a standard translator instance."""
    return GenericTranslator()
