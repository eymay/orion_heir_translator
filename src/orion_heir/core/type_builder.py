"""
Type builder for constructing HEIR types from scheme parameters.

This module provides utilities for building xDSL/HEIR types based on
FHE scheme parameters, handling the complex type system in a clean way.
"""

from typing import Dict, List, Any, Optional

from xdsl.dialects.builtin import (
    IntegerAttr, ArrayAttr, StringAttr, TensorType, 
    IntegerType, f32, Block, SSAValue
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
        
        # Application data - assuming slots equal to half the ring degree
        slots = getattr(self.scheme_params, 'slots', self.scheme_params.ring_degree/2)
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

    def create_plaintext_type_for_tensor(self, tensor_type: TensorType):
        """Create a plaintext type that matches the actual tensor being encoded."""
        from ..dialects.lwe import (
            NewLWEPlaintextType, PlaintextSpaceAttr, ApplicationDataAttr,
            InverseCanonicalEncodingAttr
        )
        from xdsl.dialects.builtin import IntegerAttr, IntegerType
        
        # Create application data that matches the actual tensor type
        app_data = ApplicationDataAttr([tensor_type])
        
        # Use the existing encoding and ring
        encoding = InverseCanonicalEncodingAttr([
            IntegerAttr(getattr(self.scheme_params, 'log_scale', 40), IntegerType(32))
        ])
        
        pt_space = PlaintextSpaceAttr([self.ring_f32, encoding])
        
        return NewLWEPlaintextType([app_data, pt_space])

    def create_plaintext_encoding(self, block: Block, constant_value: SSAValue) -> SSAValue:
        """
        Create a plaintext encoding operation for a constant.
        
        This encodes a constant tensor into the LWE plaintext space with 
        matching application data.
        """
        from ..dialects.lwe import RLWEEncodeOp, InverseCanonicalEncodingAttr
        from ..dialects.polynomial import RingAttr, PolynomialAttr
        from xdsl.dialects.builtin import f32, IntegerAttr, IntegerType
        
        # Get the actual tensor type from the constant
        tensor_type = constant_value.type
        
        # Create plaintext type that matches the tensor type
        pt_type = self.type_builder.create_plaintext_type_for_tensor(tensor_type)
        
        # Create encoding attribute with correct scale
        encoding_attr = InverseCanonicalEncodingAttr([
            IntegerAttr(getattr(self.type_builder.scheme_params, 'log_scale', 40), IntegerType(32))
        ])
        
        # Create ring attribute
        poly_attr = PolynomialAttr([
            StringAttr(f"1+x**{self.type_builder.scheme_params.ring_degree}")
        ])
        ring_attr = RingAttr([f32, poly_attr])
        
        # Create encoding operation with matching types
        encode_op = RLWEEncodeOp(
            operands=[constant_value],
            result_types=[pt_type],
            attributes={
                "encoding": encoding_attr,
                "ring": ring_attr
            }
        )
        
        block.add_op(encode_op)
        return encode_op.results[0]

    def create_padded_tensor_constant(self, block: Block, tensor_value: Any, target_slots: int) -> SSAValue:
        """Create a tensor constant padded to the target slot count."""
        from xdsl.dialects.builtin import TensorType, f32, DenseIntOrFPElementsAttr
        from xdsl.dialects.arith import ConstantOp
        import torch
        import numpy as np
        
        # Convert to numpy
        if isinstance(tensor_value, torch.Tensor):
            tensor_np = tensor_value.detach().cpu().numpy().astype(np.float32)
        else:
            tensor_np = np.array(tensor_value, dtype=np.float32).astype(np.float32)
        
        # Flatten the source data
        source_data = tensor_np.flatten()
        source_size = len(source_data)
        
        print(f"    Padding tensor from size {source_size} to {target_slots}")
        
        # Create padded data
        if source_size <= target_slots:
            # Pad with zeros
            padded_data = np.zeros(target_slots, dtype=np.float32)
            padded_data[:source_size] = source_data
        else:
            # Truncate if larger (shouldn't happen in most cases)
            padded_data = source_data[:target_slots]
            print(f"    Warning: Truncating tensor from {source_size} to {target_slots}")
        
        # Create tensor type with slot count shape
        tensor_type = TensorType(f32, [target_slots])
        float_data = [float(x) for x in padded_data]
        
        # Create constant operation
        dense_attr = DenseIntOrFPElementsAttr.create_dense_float(tensor_type, float_data)
        const_op = ConstantOp(dense_attr, tensor_type)
        block.add_op(const_op)
        
        return const_op.results[0]

    def create_slot_based_plaintext_encoding(self, block: Block, tensor_value: Any, target_scale: int = None, match_ciphertext_type: Any = None) -> SSAValue:
        """Create a plaintext encoding with slot-based padding."""
        from ..dialects.lwe import RLWEEncodeOp, InverseCanonicalEncodingAttr
        from ..dialects.polynomial import RingAttr, PolynomialAttr
        from xdsl.dialects.builtin import f32, IntegerAttr, IntegerType, StringAttr
        
        # Get slot count from scheme parameters
        slots = getattr(self.scheme_params, 'slots', 4096)
        
        # Create padded tensor constant
        padded_constant = self.create_padded_tensor_constant(block, tensor_value, slots)
        # Determine target scaling factor
        if match_ciphertext_type is not None:
            # Extract scaling factor from ciphertext type
            if hasattr(match_ciphertext_type, 'parameters') and len(match_ciphertext_type.parameters) >= 2:
                ct_pt_space = match_ciphertext_type.parameters[1]
                target_scale = ct_pt_space.encoding.scaling_factor.value.data
        elif target_scale is None:
            target_scale = getattr(self.scheme_params, 'log_scale', 40)
        
        # Use default plaintext type (which should be slot-based)
        pt_type = self.create_plaintext_type_with_scale(target_scale, match_ciphertext_type) 

        # Create encoding attributes
        encoding_attr = InverseCanonicalEncodingAttr([
            IntegerAttr(target_scale, IntegerType(32))
        ])
        
        poly_attr = PolynomialAttr([
            StringAttr(f"1+x**{self.scheme_params.ring_degree}")
        ])
        ring_attr = RingAttr([f32, poly_attr])
        
        # Create encoding operation
        encode_op = RLWEEncodeOp(
            operands=[padded_constant],
            result_types=[pt_type],
            attributes={
                "encoding": encoding_attr,
                "ring": ring_attr
            }
        )
        
        block.add_op(encode_op)
        return encode_op.results[0]
    
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
    
    def create_plaintext_type_with_scale(self, log_scale: int = None, match_ciphertext_type: Any = None):
        """Create a plaintext type with specific scaling factor or matching a ciphertext type."""
        from ..dialects.lwe import (
            NewLWEPlaintextType, PlaintextSpaceAttr,
            InverseCanonicalEncodingAttr
        )
        from xdsl.dialects.builtin import IntegerAttr, IntegerType
        
        # If matching a ciphertext type, extract its scaling factor
        if match_ciphertext_type is not None:
            if hasattr(match_ciphertext_type, 'parameters') and len(match_ciphertext_type.parameters) >= 2:
                ct_pt_space = match_ciphertext_type.parameters[1]  # PlaintextSpaceAttr
                ct_encoding = ct_pt_space.encoding
                log_scale = ct_encoding.scaling_factor.value.data
                # Also use the same application data
                self.app_data = match_ciphertext_type.parameters[0]
            else:
                app_data = self.app_data
        
        # Use provided log_scale or default
        if log_scale is None:
            log_scale = getattr(self.scheme_params, 'log_scale', 40)
        
        encoding = InverseCanonicalEncodingAttr([
            IntegerAttr(log_scale, IntegerType(32))
        ])
        
        pt_space = PlaintextSpaceAttr([self.ring_f32, encoding])
        
        return NewLWEPlaintextType([
            self.app_data,
            pt_space
        ])

    def create_ciphertext_type_with_dimension(self, dimension: int = 2, preserve_from_type: Any = None):
        """Create a ciphertext type with specific dimension, optionally preserving other attributes."""
        from ..dialects.lwe import NewLWECiphertextType, CiphertextSpaceAttr
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
        
        if preserve_from_type is not None:
            # Preserve everything except ciphertext dimension
            app_data = preserve_from_type.parameters[0]
            plaintext_space = preserve_from_type.parameters[1]  # Keep same scaling factor!
            old_ct_space = preserve_from_type.parameters[2]
            key = preserve_from_type.parameters[3]
            modulus_chain = preserve_from_type.parameters[4]
            
            # Only change the dimension
            ct_space = CiphertextSpaceAttr([
                old_ct_space.ring,
                old_ct_space.encryption_type,
                IntegerAttr(dimension, IntegerType(32))
            ])
        else:
            # Use default values
            app_data = self.app_data
            plaintext_space = self.base_pt_space
            ct_space = CiphertextSpaceAttr([
                self.ring_rns,
                StringAttr("mix"),
                IntegerAttr(dimension, IntegerType(32))
            ])
            key = self.key
            modulus_chain = self.mod_chain
        
        return NewLWECiphertextType([
            app_data,
            plaintext_space,
            ct_space,
            key,
            modulus_chain
        ])

   
    def create_relinearized_ciphertext_type(self, input_ciphertext_type: Any) -> Any:
        """Create a relinearized ciphertext type that preserves all plaintext information."""
        from ..dialects.lwe import NewLWECiphertextType, CiphertextSpaceAttr
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
        
        if not hasattr(input_ciphertext_type, 'parameters'):
            return input_ciphertext_type
        
        # Extract all components from input type
        app_data = input_ciphertext_type.parameters[0]       # ApplicationDataAttr - PRESERVE
        plaintext_space = input_ciphertext_type.parameters[1] # PlaintextSpaceAttr - PRESERVE  
        ciphertext_space = input_ciphertext_type.parameters[2] # CiphertextSpaceAttr - MODIFY SIZE ONLY
        key = input_ciphertext_type.parameters[3]            # KeyAttr - PRESERVE
        modulus_chain = input_ciphertext_type.parameters[4]  # ModulusChainAttr - PRESERVE
        
        # Create new ciphertext space with size = 2 (degree 1), preserving everything else
        new_ciphertext_space = CiphertextSpaceAttr([
            ciphertext_space.ring,           # Same ring
            ciphertext_space.encryption_type, # Same encryption type  
            IntegerAttr(2, IntegerType(32))  # Size = 2 (degree 1)
        ])
        
        # Create new ciphertext type preserving all plaintext information
        return NewLWECiphertextType([
            app_data,         # SAME application data
            plaintext_space,  # SAME plaintext space (including scaling factor!)
            new_ciphertext_space, # Only difference: size = 2
            key,             # SAME key
            modulus_chain    # SAME modulus chain
        ]) 

    def infer_result_type(self, op_type: str, lhs_type: Any, rhs_type: Any) -> Any:
        """Infer the result type for a binary operation."""
        from ..dialects.lwe import (
            NewLWEPlaintextType, NewLWECiphertextType, PlaintextSpaceAttr, CiphertextSpaceAttr,
            InverseCanonicalEncodingAttr, FullCRTPackingEncodingAttr
        )
        from xdsl.dialects.builtin import IntegerAttr, IntegerType
        
        def get_scaling_factor_from_encoding(encoding_attr):
            """Extract scaling factor from encoding attribute."""
            if isinstance(encoding_attr, InverseCanonicalEncodingAttr):
                return encoding_attr.scaling_factor.value.data
            elif isinstance(encoding_attr, FullCRTPackingEncodingAttr):
                return encoding_attr.scaling_factor.value.data
            else:
                return 0
        
        def get_encoding_from_type(type_obj):
            """Extract encoding from plaintext or ciphertext type."""
            if isinstance(type_obj, NewLWEPlaintextType):
                plaintext_space = type_obj.parameters[1]  # PlaintextSpaceAttr
                return plaintext_space.encoding
            elif isinstance(type_obj, NewLWECiphertextType):
                plaintext_space = type_obj.parameters[1]  # PlaintextSpaceAttr  
                return plaintext_space.encoding
            return None
        
        def create_result_type_with_scale_size(base_type, new_scale, new_size):
            """Create a new type with updated scaling factor."""
            if isinstance(base_type, NewLWEPlaintextType):
                # Get existing components
                app_data = base_type.parameters[0]
                plaintext_space = base_type.parameters[1]
                ring = plaintext_space.ring
                
                # Create new encoding with updated scale
                old_encoding = plaintext_space.encoding
                if isinstance(old_encoding, InverseCanonicalEncodingAttr):
                    new_encoding = InverseCanonicalEncodingAttr([
                        IntegerAttr(new_scale, IntegerType(32))
                    ])
                elif isinstance(old_encoding, FullCRTPackingEncodingAttr):
                    new_encoding = FullCRTPackingEncodingAttr([
                        IntegerAttr(new_scale, IntegerType(32))
                    ])
                else:
                    new_encoding = old_encoding
                
                # Create new plaintext space
                new_pt_space = PlaintextSpaceAttr([ring, new_encoding])
                
                # Return new plaintext type
                return NewLWEPlaintextType([app_data, new_pt_space])
                
            elif isinstance(base_type, NewLWECiphertextType):
                # Get existing components
                app_data = base_type.parameters[0]
                plaintext_space = base_type.parameters[1]
                ciphertext_space = base_type.parameters[2]
                key = base_type.parameters[3]
                modulus_chain = base_type.parameters[4]
                ring = plaintext_space.ring
                
                # Create new encoding with updated scale
                old_encoding = plaintext_space.encoding
                if isinstance(old_encoding, InverseCanonicalEncodingAttr):
                    new_encoding = InverseCanonicalEncodingAttr([
                        IntegerAttr(new_scale, IntegerType(32))
                    ])
                elif isinstance(old_encoding, FullCRTPackingEncodingAttr):
                    new_encoding = FullCRTPackingEncodingAttr([
                        IntegerAttr(new_scale, IntegerType(32))
                    ])
                else:
                    new_encoding = old_encoding
                
                # Create new plaintext space
                new_pt_space = PlaintextSpaceAttr([ring, new_encoding])
                new_ct_space = CiphertextSpaceAttr([
                                ciphertext_space.ring,
                                ciphertext_space.encryption_type,
                                IntegerAttr(result_size, IntegerType(32))  # dimension
                            ])
                # Return new ciphertext type
                return NewLWECiphertextType([
                    app_data, new_pt_space, new_ct_space, key, modulus_chain
                ])
            
            return base_type
        
        # Extract encodings from both types
        lhs_encoding = get_encoding_from_type(lhs_type)
        rhs_encoding = get_encoding_from_type(rhs_type)
        
        if lhs_encoding is None or rhs_encoding is None:
            # If we can't extract encodings, return LHS type
            return lhs_type
        
        # Get scaling factors
        lhs_scale = get_scaling_factor_from_encoding(lhs_encoding)
        rhs_scale = get_scaling_factor_from_encoding(rhs_encoding)
        
        # Get plaintext modulus for FullCRT operations
        plaintext_modulus = getattr(self.scheme_params, 'plaintext_modulus', 65537)
        
        result_size = 0
        # Compute result scaling factor based on operation and encoding types
        if op_type == 'mul':
            result_size = lhs_type.ciphertext_space.size.value.data + rhs_type.ciphertext_space.size.value.data -1

            # Handle multiplication scaling
            if isinstance(lhs_encoding, FullCRTPackingEncodingAttr):
                # For FullCRT: (xScale * yScale) % plaintextModulus
                result_scale = (lhs_scale * rhs_scale) % plaintext_modulus
            elif isinstance(lhs_encoding, InverseCanonicalEncodingAttr):
                # For InverseCanonical: xScale + yScale
                result_scale = lhs_scale + rhs_scale
            else:
                # Default case
                result_scale = lhs_scale
                
        elif op_type in ['add', 'sub']:
            # Addition/subtraction preserves the larger scale
            result_scale = max(lhs_scale, rhs_scale)
            result_size = max(lhs_type.ciphertext_space.size.value.data, rhs_type.ciphertext_space.size.value.data)
            
        elif op_type == 'rotate':
            result_scale = lhs_scale
            result_size = 2
        else:
            # Default: preserve LHS scale
            result_scale = lhs_scale
        
        # Create result type with computed scale
        return create_result_type_with_scale_size(lhs_type, result_scale, result_size)

    def infer_result_type_with_relinearization(self, op_type: str, lhs_type: Any, rhs_type: Any) -> Any:
        """Infer the result type for a binary operation, handling dimension changes correctly."""
        from ..dialects.lwe import NewLWECiphertextType, CiphertextSpaceAttr, PlaintextSpaceAttr, InverseCanonicalEncodingAttr
        from xdsl.dialects.builtin import IntegerAttr, IntegerType, StringAttr
        
        # For multiplication operations, dimension increases BUT scaling factor should be computed correctly
        if op_type == 'mul':
            if isinstance(lhs_type, NewLWECiphertextType):
                # Get current dimension from ciphertext space
                ct_space = lhs_type.parameters[2]  # CiphertextSpaceAttr
                current_dim = ct_space.size.value.data
                
                # After multiplication, dimension increases by 1
                new_dim = current_dim + 1
                
                # Extract scaling factors and compute result scaling factor
                lhs_pt_space = lhs_type.parameters[1]  # PlaintextSpaceAttr
                lhs_encoding = lhs_pt_space.encoding
                lhs_scale = lhs_encoding.scaling_factor.value.data
                
                # For self-multiplication or when rhs_type is same as lhs_type
                if rhs_type == lhs_type or not hasattr(rhs_type, 'parameters'):
                    # Self-multiplication: result scale = 2 * input_scale (in log domain: scale + scale)
                    result_scale = lhs_scale + lhs_scale
                else:
                    # Different operands
                    rhs_pt_space = rhs_type.parameters[1]
                    rhs_encoding = rhs_pt_space.encoding  
                    rhs_scale = rhs_encoding.scaling_factor.value.data
                    result_scale = lhs_scale + rhs_scale
                
                # Create new encoding with computed scaling factor
                new_encoding = InverseCanonicalEncodingAttr([
                    IntegerAttr(result_scale, IntegerType(32))
                ])
                
                # Create new plaintext space with updated encoding
                new_pt_space = PlaintextSpaceAttr([lhs_pt_space.ring, new_encoding])
                
                # Create new ciphertext space with increased dimension
                new_ct_space = CiphertextSpaceAttr([
                    ct_space.ring,
                    ct_space.encryption_type,
                    IntegerAttr(new_dim, IntegerType(32))
                ])
                
                # Create new ciphertext type with updated scaling factor and dimension
                return NewLWECiphertextType([
                    lhs_type.parameters[0],  # app_data
                    new_pt_space,            # updated plaintext_space
                    new_ct_space,            # updated ciphertext_space  
                    lhs_type.parameters[3],  # key
                    lhs_type.parameters[4]   # modulus_chain
                ])
        
        # For other operations, preserve the type
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
