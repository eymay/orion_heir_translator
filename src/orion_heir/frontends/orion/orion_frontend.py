"""
Orion frontend for the HEIR translator.

This module provides Orion-specific functionality for extracting operations
and scheme parameters from Orion FHE computations.
"""

from typing import List, Any, Dict, Optional, Union
import yaml
import torch
from pathlib import Path

from ...core.types import FHEOperation, FrontendInterface, SchemeParameters
from .scheme_params import OrionSchemeParameters


class OrionFrontend(FrontendInterface):
    """
    Frontend for translating Orion FHE operations to HEIR.
    
    This class implements the FrontendInterface for Orion,
    providing Orion-specific logic for operation extraction
    and parameter handling from compiled Orion models.
    """
    
    def __init__(self):
        """Initialize the Orion frontend with supported operations."""
        self._supported_operations = {
            # CKKS arithmetic operations
            'add': {
                'description': 'Add two ciphertexts',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'additive'
            },
            'sub': {
                'description': 'Subtract two ciphertexts',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'additive'
            },
            'mul': {
                'description': 'Multiply two ciphertexts',
                'operands': 2,
                'level_change': -1,
                'noise_growth': 'multiplicative'
            },
            'negate': {
                'description': 'Negate a ciphertext',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            
            # CKKS plaintext operations
            'add_plain': {
                'description': 'Add plaintext to ciphertext',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            'sub_plain': {
                'description': 'Subtract plaintext from ciphertext',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            'mul_plain': {
                'description': 'Multiply ciphertext by plaintext',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'linear'
            },
            
            # CKKS rotation operations
            'rotate': {
                'description': 'Rotate ciphertext slots',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            
            # Primary Orion operation
            'linear_transform': {
                'description': 'Orion linear transform using precomputed diagonals',
                'operands': 1,
                'level_change': -1,
                'noise_growth': 'complex'
            },
            
            # Noise management
            'rescale': {
                'description': 'Rescale ciphertext to manage noise',
                'operands': 1,
                'level_change': -1,
                'noise_growth': 'reduction'
            },
            'relinearize': {
                'description': 'Reduce ciphertext size after multiplication',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            'bootstrap': {
                'description': 'Refresh ciphertext (reset noise and level)',
                'operands': 1,
                'level_change': 'reset',
                'noise_growth': 'reset'
            },
            
            # Encoding operations
            'encode': {
                'description': 'Encode plaintext for CKKS',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'none'
            },
            'decode': {
                'description': 'Decode CKKS plaintext',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'none'
            },
            'encrypt': {
                'description': 'Encrypt plaintext to ciphertext',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'initial'
            },
            'decrypt': {
                'description': 'Decrypt ciphertext to plaintext',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'none'
            }
        }
    
    def extract_operations(self, source: Any) -> List[FHEOperation]:
        """
        Extract FHE operations from various Orion sources.
        
        Args:
            source: Can be:
                - Compiled Orion model (after orion.compile())
                - List of operation dictionaries
                
        Returns:
            List of FHEOperation objects
            
        Raises:
            ValueError: If source type is not supported
        """
        if self._is_compiled_orion_model(source):
            return self._extract_from_compiled_model(source)
        elif isinstance(source, (list, tuple)):
            return self._extract_from_operation_list(source)
        elif hasattr(source, 'operations'):
            return self._extract_from_operation_list(source.operations)
        else:
            raise ValueError(
                f"Unsupported source type: {type(source)}. "
                f"Expected compiled Orion model or list of operations."
            )
    
    def _is_compiled_orion_model(self, source: Any) -> bool:
        """
        Check if source is a compiled Orion model.
        
        A compiled Orion model should have been through orion.compile()
        and have Orion-specific attributes.
        """
        if not hasattr(source, '__class__'):
            return False
            
        # Check if it's from an Orion module
        module_name = getattr(source.__class__, '__module__', '')
        if 'orion' in module_name.lower():
            return True
            
        # Check for Orion-specific attributes that appear after compilation
        orion_indicators = [
            'fc1', 'linear', 'conv',  # Common layer names
            'he_mode',                # Orion-specific mode
            'scheme',                 # Orion scheme reference
        ]
        
        return any(hasattr(source, attr) for attr in orion_indicators)
    
    def _extract_from_compiled_model(self, model: Any) -> List[FHEOperation]:
        """
        Extract operations from a compiled Orion model.
        
        This walks through the model's layers and emits the corresponding
        CKKS operations that would be performed during HE inference.
        """
        operations = []
        
        # Walk through all modules to find Orion layers
        for layer_name, layer in model.named_modules():
            if self._is_orion_layer(layer):
                layer_operations = self._get_layer_operations(layer, layer_name)
                operations.extend(layer_operations)
        
        if not operations:
            raise ValueError(
                "No Orion layers found in model. "
                "Make sure the model has been compiled with orion.compile()."
            )
        
        return operations
    
    def _is_orion_layer(self, layer: Any) -> bool:
        """
        Check if a layer is an Orion layer that performs FHE operations.
        
        Orion layers typically have specific attributes set during compilation.
        """
        if not hasattr(layer, '__class__'):
            return False
            
        # Check layer type
        layer_type = layer.__class__.__name__
        orion_layer_types = ['Linear', 'Conv2d', 'LinearTransform', 'Activation']
        
        if layer_type in orion_layer_types:
            # Additional check for Orion-specific compilation artifacts
            compilation_indicators = [
                'diagonals',      # Linear transform diagonals
                'transform_ids',  # Backend transform IDs
                'on_weight',      # Orion weight copy
                'on_bias',        # Orion bias copy
                'level',          # Assigned level
            ]
            
            return any(hasattr(layer, attr) for attr in compilation_indicators)
        
        return False
    
    def _get_layer_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get the CKKS operations for a specific compiled Orion layer.
        
        This emits the operations that correspond to what the layer
        would do during HE inference, using its compiled artifacts.
        """
        layer_type = layer.__class__.__name__
        
        if layer_type == 'Linear':
            return self._get_linear_operations(layer, layer_name)
        elif layer_type == 'Conv2d':
            return self._get_conv_operations(layer, layer_name)
        elif layer_type in ['Activation', 'Quad', 'Chebyshev']:
            return self._get_activation_operations(layer, layer_name)
        else:
            # Generic layer - try to infer operations
            return self._get_generic_operations(layer, layer_name)
    
    def _get_linear_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for a Linear layer.
        
        Linear layers in Orion use precomputed diagonal transforms.
        """
        operations = []
        level = getattr(layer, 'level', 1)
        
        # Primary operation: linear transform using Orion's precomputed diagonals
        if hasattr(layer, 'transform_ids') and layer.transform_ids:
            operations.append(FHEOperation(
                op_type="linear_transform",
                method_name="linear_transform",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_linear",
                level=level,
                metadata={
                    'operation': 'orion_linear_transform',
                    'layer': layer_name,
                    'layer_type': 'Linear',
                    'transform_blocks': len(layer.transform_ids),
                    'diagonal_count': len(layer.diagonals) if hasattr(layer, 'diagonals') else 0
                }
            ))
        
        # Output rotations if needed (SIMD packing optimizations)
        if hasattr(layer, 'output_rotations') and layer.output_rotations > 0:
            for i in range(layer.output_rotations):
                # Rotation
                operations.append(FHEOperation(
                    op_type="rotate",
                    method_name="rotate",
                    args=[2**i],
                    kwargs={"offset": 2**i},
                    result_var=f"{layer_name}_rot_{i}",
                    level=level,
                    metadata={
                        'operation': 'output_rotation',
                        'layer': layer_name,
                        'rotation_index': i
                    }
                ))
                
                # Accumulation
                operations.append(FHEOperation(
                    op_type="add",
                    method_name="add",
                    args=[],
                    kwargs={},
                    result_var=f"{layer_name}_acc_{i}",
                    level=level,
                    metadata={
                        'operation': 'rotation_accumulation',
                        'layer': layer_name,
                        'rotation_index': i
                    }
                ))
        
        # Bias addition using Orion's precomputed bias plaintext
        if hasattr(layer, 'bias') and layer.bias is not None:
            operations.append(FHEOperation(
                op_type="add_plain",
                method_name="add_plain",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_result",
                level=level,
                metadata={
                    'operation': 'bias_addition',
                    'layer': layer_name,
                    'layer_type': 'Linear'
                }
            ))
        
        return operations
    
    def _get_conv_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for a Conv2d layer.
        
        Convolution in Orion is implemented using linear transforms
        with Toeplitz matrices converted to diagonals.
        """
        operations = []
        level = getattr(layer, 'level', 1)
        
        # Convolution as linear transform
        if hasattr(layer, 'transform_ids') and layer.transform_ids:
            operations.append(FHEOperation(
                op_type="linear_transform",
                method_name="linear_transform",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_conv",
                level=level,
                metadata={
                    'operation': 'orion_convolution',
                    'layer': layer_name,
                    'layer_type': 'Conv2d',
                    'transform_blocks': len(layer.transform_ids)
                }
            ))
        
        # Bias addition if present
        if hasattr(layer, 'bias') and layer.bias is not None:
            operations.append(FHEOperation(
                op_type="add_plain",
                method_name="add_plain",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_result",
                level=level,
                metadata={
                    'operation': 'bias_addition',
                    'layer': layer_name,
                    'layer_type': 'Conv2d'
                }
            ))
        
        return operations
    
    def _get_activation_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for activation layers.
        
        Activation functions in FHE are polynomial approximations.
        """
        operations = []
        level = getattr(layer, 'level', 1)
        layer_type = layer.__class__.__name__
        
        if layer_type == 'Quad':
            # Quadratic activation: x^2
            operations.append(FHEOperation(
                op_type="mul",
                method_name="mul",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_quad",
                level=level - 1,
                metadata={
                    'operation': 'quadratic_activation',
                    'layer': layer_name,
                    'layer_type': layer_type
                }
            ))
        else:
            # General polynomial activation (would need polynomial dialect)
            # For now, represent as a placeholder
            operations.append(FHEOperation(
                op_type="mul",  # Simplified representation
                method_name="mul",
                args=[],
                kwargs={},
                result_var=f"{layer_name}_activation",
                level=level - 1,
                metadata={
                    'operation': 'polynomial_activation',
                    'layer': layer_name,
                    'layer_type': layer_type
                }
            ))
        
        return operations
    
    def _get_generic_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for generic/unknown layer types.
        """
        return []
    
    def _extract_from_operation_list(self, operations: List[Any]) -> List[FHEOperation]:
        """
        Extract operations from a list of operation objects or dictionaries.
        """
        fhe_operations = []
        
        for op in operations:
            if isinstance(op, FHEOperation):
                fhe_operations.append(op)
            elif isinstance(op, dict):
                fhe_op = self._dict_to_fhe_operation(op)
                if fhe_op:
                    fhe_operations.append(fhe_op)
            else:
                fhe_op = self._object_to_fhe_operation(op)
                if fhe_op:
                    fhe_operations.append(fhe_op)
        
        return fhe_operations
    
    def _dict_to_fhe_operation(self, op_dict: Dict[str, Any]) -> Optional[FHEOperation]:
        """Convert a dictionary to an FHEOperation."""
        try:
            return FHEOperation(
                op_type=op_dict.get('op_type', 'unknown'),
                method_name=op_dict.get('method_name', op_dict.get('op_type', 'unknown')),
                args=op_dict.get('args', []),
                kwargs=op_dict.get('kwargs', {}),
                result_var=op_dict.get('result_var'),
                level=op_dict.get('level'),
                metadata=op_dict.get('metadata', {})
            )
        except Exception:
            return None
    
    def _object_to_fhe_operation(self, op_obj: Any) -> Optional[FHEOperation]:
        """Convert an object to an FHEOperation."""
        if not hasattr(op_obj, 'op_type'):
            return None
            
        try:
            return FHEOperation(
                op_type=getattr(op_obj, 'op_type', 'unknown'),
                method_name=getattr(op_obj, 'method_name', getattr(op_obj, 'op_type', 'unknown')),
                args=getattr(op_obj, 'args', []),
                kwargs=getattr(op_obj, 'kwargs', {}),
                result_var=getattr(op_obj, 'result_var', None),
                level=getattr(op_obj, 'level', None),
                metadata=getattr(op_obj, 'metadata', {})
            )
        except Exception:
            return None
    
    def get_supported_operations(self) -> List[str]:
        """Get list of all CKKS operations supported by this frontend."""
        return list(self._supported_operations.keys())
    
    def get_operation_info(self, op_type: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific operation type."""
        return self._supported_operations.get(op_type)
    
    def extract_scheme_parameters(self, source: Union[str, Path, Dict[str, Any], Any]) -> SchemeParameters:
        """
        Extract scheme parameters from various sources.
        
        Args:
            source: Configuration file path, dict, or Orion scheme object
            
        Returns:
            OrionSchemeParameters object
        """
        if isinstance(source, (str, Path)):
            return self._load_scheme_from_config(Path(source))
        elif isinstance(source, dict):
            return self._create_scheme_from_config(source)
        elif hasattr(source, 'logN') or hasattr(source, 'params'):
            return self._create_scheme_from_orion_object(source)
        else:
            return self._create_default_scheme()
    
    def _load_scheme_from_config(self, config_path: Path) -> OrionSchemeParameters:
        """Load scheme parameters from YAML configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return self._create_scheme_from_config(config)
    
    def _create_scheme_from_config(self, config: Dict[str, Any]) -> OrionSchemeParameters:
        """Create scheme parameters from configuration dictionary."""
        ckks_params = config.get('ckks_params', {})
        orion_params = config.get('orion', {})
        
        return OrionSchemeParameters(
            logN=ckks_params.get('LogN', 13),
            logQ=ckks_params.get('LogQ', [55, 45, 45, 55]),
            logP=ckks_params.get('LogP', [55]),
            logScale=ckks_params.get('LogScale', 45),
            slots=ckks_params.get('slots', 2**12),
            ring_degree=2**ckks_params.get('LogN', 13),
            backend=orion_params.get('backend', 'lattigo')
        )
    
    def _create_scheme_from_orion_object(self, orion_obj: Any) -> OrionSchemeParameters:
        """Create scheme parameters from Orion scheme object."""
        if hasattr(orion_obj, 'params'):
            params = orion_obj.params
            return OrionSchemeParameters(
                logN=getattr(params, 'logN', 13),
                logQ=getattr(params, 'logQ', [55, 45, 45, 55]),
                logP=getattr(params, 'logP', [55]),
                logScale=getattr(params, 'logScale', 45),
                slots=getattr(params, 'slots', 2**12),
                ring_degree=getattr(params, 'ring_degree', 2**13),
                backend=getattr(params, 'backend', 'lattigo')
            )
        else:
            return OrionSchemeParameters(
                logN=getattr(orion_obj, 'logN', 13),
                logQ=getattr(orion_obj, 'logQ', [55, 45, 45, 55]),
                logP=getattr(orion_obj, 'logP', [55]),
                logScale=getattr(orion_obj, 'logScale', 45),
                slots=getattr(orion_obj, 'slots', 2**12),
                ring_degree=getattr(orion_obj, 'ring_degree', 2**13),
                backend=getattr(orion_obj, 'backend', 'lattigo')
            )
    
    def _create_default_scheme(self) -> OrionSchemeParameters:
        """Create default scheme parameters for testing."""
        return OrionSchemeParameters(
            logN=13,
            logQ=[55, 45, 45, 55],
            logP=[55],
            logScale=45,
            slots=2**12,
            ring_degree=2**13,
            backend='lattigo'
        )


def create_orion_frontend() -> OrionFrontend:
    """Factory function to create an Orion frontend."""
    return OrionFrontend()
