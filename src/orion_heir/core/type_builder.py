"""
Type builder for constructing HEIR types from scheme parameters.

This module provides utilities for building xDSL/HEIR types based on
FHE scheme parameters, handling the complex type system in a clean way.
"""

from typing import Dict, List, Any, Optional

from xdsl.dialects.builtin import (
    IntegerAttr, ArrayAttr, StringAttr, TensorType, 
    IntegerType, f32
)

from .types import SchemeParameters


class TypeBuilder:
    """
    Builder for HEIR types based on FHE scheme parameters.
    
    This class encapsulates the complex logic for creating proper
    HEIR types from scheme parameters.
    """
    
    def __init__(self, scheme_params: SchemeParameters):
        self.scheme_params = scheme_params
        self._setup_base_types()
    
    def _setup_base_types(self):
        """Setup base types and attributes."""
        from ..dialects.mod_arith import ModArithType
        from ..dialects.rns import RNSType
        from ..dialects.polynomial import PolynomialAttr, RingAttr
        from ..dialects.lwe import (
            InverseCanonicalEncodingAttr, PlaintextSpaceAttr,
            CiphertextSpaceAttr, ApplicationDataAttr, KeyAttr,
            ModulusChainAttr
        )
        from ..dialects.ckks import SchemeParamAttr
        
        # Build modular arithmetic types
        moduli = self.scheme_params.ciphertext_modulus_chain
        self.mod_types = [
            ModArithType([IntegerAttr(modulus, IntegerType(64))])
            for modulus in moduli
        ]
        
        # Build RNS type
        self.rns_type = RNSType([ArrayAttr(self.mod_types)])
        
        # Build polynomial attribute
        self.poly_attr = PolynomialAttr([
            StringAttr(f"1 + x**{self.scheme_params.ring_degree}")
        ])
        
        # Build rings
        self.ring_rns = RingAttr([self.rns_type, self.poly_attr])
        self.ring_f32 = RingAttr([f32, self.poly_attr])
        
        # Build LWE attributes
        self.base_encoding = InverseCanonicalEncodingAttr([
            IntegerAttr(getattr(self.scheme_params, 'log_scale', 40), IntegerType(32))
        ])
        
        self.key = KeyAttr([])
        
        # Application data - assuming slots equal to ring degree for now
        slots = getattr(self.scheme_params, 'slots', self.scheme_params.ring_degree)
        self.app_data = ApplicationDataAttr([TensorType(f32, [slots])])
        
        # Plaintext space
        self.base_pt_space = PlaintextSpaceAttr([self.ring_f32, self.base_encoding])
        
        # Ciphertext space
        self.ct_space = CiphertextSpaceAttr([
            self.ring_rns,
            StringAttr("mix"),  # encryption type
            IntegerAttr(2, IntegerType(32))  # dimension
        ])
        
        # Modulus chain
        self.mod_chain = ModulusChainAttr([
            ArrayAttr([IntegerAttr(mod, IntegerType(64)) for mod in moduli]),
            IntegerAttr(len(moduli) - 1, IntegerType(32))  # current level
        ])
        
        # CKKS scheme parameters
        self.scheme_param = SchemeParamAttr([
            IntegerAttr(getattr(self.scheme_params, 'log_n', 13), IntegerType(32)),
            ArrayAttr([IntegerAttr(mod, IntegerType(64)) for mod in moduli]),
            ArrayAttr([IntegerAttr(getattr(self.scheme_params, 'plaintext_modulus', 65537), IntegerType(64))]),
            IntegerAttr(getattr(self.scheme_params, 'log_scale', 40), IntegerType(32))
        ])
    
    def get_default_ciphertext_type(self):
        """Get the default ciphertext type at maximum level."""
        from ..dialects.lwe import NewLWECiphertextType
        
        return NewLWECiphertextType([
            self.app_data,
            self.base_pt_space,
            self.ct_space,
            self.key,
            self.mod_chain
        ])
    
    def get_default_plaintext_type(self):
        """Get the default plaintext type."""
        from ..dialects.lwe import NewLWEPlaintextType
        
        return NewLWEPlaintextType([
            self.app_data,
            self.base_pt_space
        ])
    
    def create_ciphertext_type_at_level(self, level: int):
        """Create a ciphertext type at a specific level."""
        from ..dialects.lwe import NewLWECiphertextType, ModulusChainAttr
        
        # Create modulus chain for this level
        moduli = self.scheme_params.ciphertext_modulus_chain[:level + 1]
        level_mod_chain = ModulusChainAttr([
            ArrayAttr([IntegerAttr(mod, IntegerType(64)) for mod in moduli]),
            IntegerAttr(level, IntegerType(32))
        ])
        
        return NewLWECiphertextType([
            self.app_data,
            self.base_pt_space,
            self.ct_space,
            self.key,
            level_mod_chain
        ])
    
    def create_plaintext_type_with_scale(self, log_scale: int):
        """Create a plaintext type with specific scaling factor."""
        from ..dialects.lwe import (
            NewLWEPlaintextType, PlaintextSpaceAttr,
            InverseCanonicalEncodingAttr
        )
        
        encoding = InverseCanonicalEncodingAttr([
            IntegerAttr(log_scale, IntegerType(32))
        ])
        
        pt_space = PlaintextSpaceAttr([self.ring_f32, encoding])
        
        return NewLWEPlaintextType([
            self.app_data,
            pt_space
        ])
    
    def infer_result_type(self, op_type: str, lhs_type: Any, rhs_type: Any) -> Any:
        """Infer the result type for a binary operation."""
        # For most operations, result type is same as LHS
        # In a more sophisticated implementation, this would handle
        # level changes, scale changes, etc.
        
        if op_type == 'mul':
            # Multiplication typically reduces level
            # For now, return same type
            return lhs_type
        elif op_type in ['add', 'sub']:
            # Addition/subtraction preserves level
            return lhs_type
        else:
            return lhs_type
    
    def infer_plaintext_result_type(self, op_type: str, ct_type: Any, pt_type: Any) -> Any:
        """Infer the result type for ciphertext-plaintext operations."""
        # Result is typically a ciphertext of the same type
        return ct_type
    
    def create_module_attributes(self) -> Dict[str, Any]:
        """Create module-level attributes."""
        return {
            "ckks.schemeParam": self.scheme_param
        }
