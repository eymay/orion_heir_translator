"""
Clean Orion frontend with hardcoded CKKS operations.

This module provides the Orion frontend with built-in knowledge of all
CKKS operations that Orion supports, including the new matmul operation.
"""

from typing import List, Any, Dict, Optional
import torch
import yaml
from pathlib import Path

from ...core.types import FHEOperation, FrontendInterface, SchemeParameters
from .scheme_params import OrionSchemeParameters


class OrionFrontend(FrontendInterface):
    """
    Clean Orion frontend with hardcoded knowledge of CKKS operations.
    
    This frontend contains built-in knowledge of all operations that
    Orion supports, based on the CKKS homomorphic encryption scheme.
    No templates or complex abstractions - just pure CKKS operations.
    """
    
    def __init__(self):
        # Hardcoded knowledge of Orion's CKKS operations
        self._orion_operations = self._initialize_orion_operations()
    
    def _initialize_orion_operations(self) -> Dict[str, Dict]:
        """Initialize hardcoded knowledge of Orion's supported operations."""
        return {
            # Core CKKS arithmetic operations
            'add': {
                'description': 'Homomorphic addition of two ciphertexts',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'additive'
            },
            'mul': {
                'description': 'Homomorphic multiplication of two ciphertexts', 
                'operands': 2,
                'level_change': -1,  # Consumes one level
                'noise_growth': 'multiplicative'
            },
            'rotate': {
                'description': 'Cyclic rotation of SIMD slots',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'minimal',
                'parameters': ['offset']
            },
            
            # Plaintext operations
            'mul_plain': {
                'description': 'Multiply ciphertext with plaintext',
                'operands': 2,  # ciphertext + plaintext
                'level_change': 0,
                'noise_growth': 'multiplicative_plain'
            },
            'add_plain': {
                'description': 'Add plaintext to ciphertext',
                'operands': 2,  # ciphertext + plaintext  
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            'sub_plain': {
                'description': 'Subtract plaintext from ciphertext',
                'operands': 2,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            
            # Matrix multiplication operation (high-level)
            'matmul': {
                'description': 'Matrix multiplication (high-level operation)',
                'operands': 2,  # ciphertext + plaintext
                'level_change': 0,  # Will be determined by lowering pass
                'noise_growth': 'depends_on_lowering'
            },
            
            # Noise management operations
            'rescale': {
                'description': 'Rescale ciphertext to manage noise',
                'operands': 1,
                'level_change': -1,  # Moves to lower level
                'noise_growth': 'reduction'
            },
            'relinearize': {
                'description': 'Reduce ciphertext size after multiplication',
                'operands': 1,
                'level_change': 0,
                'noise_growth': 'minimal'
            },
            'bootstrap': {
                'description': 'Refresh ciphertext to enable more operations',
                'operands': 1,
                'level_change': 'reset',  # Resets to highest level
                'noise_growth': 'reset'
            },
            
            # Encoding/decoding operations  
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
    
    def get_supported_operations(self) -> List[str]:
        """Get list of all operations Orion supports."""
        return list(self._orion_operations.keys())
    
    def get_operation_info(self, op_type: str) -> Optional[Dict]:
        """Get information about a specific operation."""
        return self._orion_operations.get(op_type)
    
    def create_simple_test_operations(self) -> List[FHEOperation]:
        """Create simple test operations for validation."""
        operations = []
        
        # Input encoding
        operations.append(FHEOperation(
            op_type="encode",
            method_name="rlwe_encode",
            args=[torch.tensor([1.0, 2.0, 3.0, 4.0])],
            kwargs={},
            result_var="input_encoded",
            level=2,
            metadata={'operation': 'encode_input'}
        ))
        
        # Encode weight matrix
        operations.append(FHEOperation(
            op_type="encode",
            method_name="rlwe_encode",
            args=[torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]])],
            kwargs={},
            result_var="weight_encoded",
            level=2,
            metadata={'operation': 'encode_weight'}
        ))
        
        # Matrix multiplication (now a first-class CKKS operation)
        operations.append(FHEOperation(
            op_type="matmul",
            method_name="matmul",
            args=[],
            kwargs={},
            result_var="matmul_result",
            level=2,
            metadata={'operation': 'matrix_multiplication', 'plaintext_input': 'weight_encoded'}
        ))
        
        return operations
    
    def create_basic_operation(self, op_type: str, **kwargs) -> Optional[FHEOperation]:
        """Create a basic CKKS operation if it's supported by Orion."""
        if op_type not in self._orion_operations:
            return None
        
        op_info = self._orion_operations[op_type]
        
        return FHEOperation(
            op_type=op_type,
            method_name=op_type,
            args=kwargs.get('args', []),
            kwargs=kwargs.get('operation_kwargs', {}), 
            result_var=kwargs.get('result_var', f'{op_type}_result'),
            level=kwargs.get('level', 2),
            metadata={
                'operation': op_type,
                'description': op_info['description']
            }
        )
    
    # FrontendInterface implementation
    def extract_operations(self, source: Any) -> List[FHEOperation]:
        """
        Extract operations from source.
        Since we have minimal hardcoded operations, this is simple.
        """
        if isinstance(source, str):
            if source == 'test':
                return self.create_simple_test_operations()
            else:
                return []
        
        elif isinstance(source, list):
            # Convert list of dicts to FHE operations
            operations = []
            for item in source:
                if isinstance(item, FHEOperation):
                    operations.append(item)
                elif isinstance(item, dict):
                    op = FHEOperation(
                        op_type=item.get('op_type', 'unknown'),
                        method_name=item.get('method_name', item.get('op_type', 'unknown')),
                        args=item.get('args', []),
                        kwargs=item.get('kwargs', {}),
                        result_var=item.get('result_var'),
                        level=item.get('level'),
                        metadata=item.get('metadata', {})
                    )
                    operations.append(op)
            return operations
        
        else:
            # Default: return simple test operations
            return self.create_simple_test_operations()
    
    def extract_scheme_parameters(self, source: Any) -> SchemeParameters:
        """Extract scheme parameters from source."""
        if isinstance(source, (str, Path)):
            # Config file path
            return self._load_scheme_from_config(source)
        elif isinstance(source, dict):
            # Config dictionary
            return self._create_scheme_from_config(source)
        elif hasattr(source, 'logN'):
            # Orion scheme object
            return self._create_scheme_from_orion(source)
        else:
            # Use default parameters
            return self._create_default_scheme()
    
    def _load_scheme_from_config(self, config_path: Path) -> OrionSchemeParameters:
        """Load scheme parameters from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return self._create_scheme_from_config(config)
    
    def _create_scheme_from_config(self, config: Dict[str, Any]) -> OrionSchemeParameters:
        """Create scheme parameters from config dictionary."""
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
    
    def _create_scheme_from_orion(self, orion_scheme: Any) -> OrionSchemeParameters:
        """Create scheme parameters from Orion scheme object."""
        return OrionSchemeParameters(
            logN=getattr(orion_scheme, 'logN', 13),
            logQ=getattr(orion_scheme, 'logQ', [55, 45, 45, 55]),
            logP=getattr(orion_scheme, 'logP', [55]),
            logScale=getattr(orion_scheme, 'logScale', 45),
            slots=getattr(orion_scheme, 'slots', 2**12),
            ring_degree=getattr(orion_scheme, 'ring_degree', 2**13),
            backend=getattr(orion_scheme, 'backend', 'lattigo')
        )
    
    def _create_default_scheme(self) -> OrionSchemeParameters:
        """Create default Orion scheme parameters."""
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


def list_orion_operations():
    """List all operations that Orion supports."""
    frontend = OrionFrontend()
    
    print("Orion Supported Operations:")
    print("=" * 40)
    
    for op_type in frontend.get_supported_operations():
        info = frontend.get_operation_info(op_type)
        print(f"{op_type:12} - {info['description']}")
        print(f"{'':12}   Operands: {info['operands']}, Level change: {info['level_change']}")
    
    print(f"\n✅ All operations are first-class CKKS operations")
    print(f"✅ matmul will be lowered by a separate pass")
    print(f"✅ No templates or abstractions needed")


if __name__ == "__main__":
    # Demo the clean frontend
    print("🚀 Clean Orion Frontend Demo")
    print("=" * 40)
    
    list_orion_operations()
    
    frontend = OrionFrontend()
    
    print(f"\n📊 Testing simple operations:")
    test_ops = frontend.create_simple_test_operations()
    print(f"Generated {len(test_ops)} operations:")
    for op in test_ops:
        print(f"  • {op.op_type} -> {op.result_var}")
    
    print(f"\n✅ Frontend is now minimal and clean!")
    print(f"✅ No complex abstractions or templates!")
    print(f"✅ Just pure CKKS operations + matmul!")
