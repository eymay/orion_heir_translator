"""
Orion frontend for the HEIR translator.

This module provides Orion-specific functionality for extracting operations
and scheme parameters from Orion FHE computations.
"""

from typing import List, Any, Dict, Optional
import yaml
from pathlib import Path

from ...core.types import FHEOperation, FrontendInterface, SchemeParameters
from .scheme_params import OrionSchemeParameters
from .operation_extractor import OrionOperationExtractor


class OrionFrontend(FrontendInterface):
    """
    Frontend for translating Orion FHE operations to HEIR.
    
    This class implements the FrontendInterface for Orion,
    providing Orion-specific logic for operation extraction
    and parameter handling.
    """
    
    def __init__(self):
        self.operation_extractor = OrionOperationExtractor()
    
    def extract_operations(self, source: Any) -> List[FHEOperation]:
        """
        Extract FHE operations from Orion source.
        
        Args:
            source: Can be a list of operations, trace data, or other Orion artifacts
            
        Returns:
            List of generic FHE operations
        """
        if isinstance(source, list):
            # Direct list of operations
            return self.operation_extractor.convert_operations(source)
        elif hasattr(source, 'operations'):
            # Object with operations attribute
            return self.operation_extractor.convert_operations(source.operations)
        else:
            # Try to extract from other formats
            return self.operation_extractor.extract_from_source(source)
    
    def extract_scheme_parameters(self, source: Any) -> SchemeParameters:
        """
        Extract scheme parameters from Orion source.
        
        Args:
            source: Orion configuration, scheme object, or config file path
            
        Returns:
            Scheme parameters object
        """
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
            logN=ckks_params.get('LogN', [13]),
            logQ=ckks_params.get('LogQ', [55, 45, 45, 55]),
            logP=ckks_params.get('LogP', [55]),
            logScale=ckks_params.get('LogScale', 45),
            slots=ckks_params.get('slots', 2**12),
            ring_degree=2**ckks_params.get('LogN', [13])[0],
            backend=orion_params.get('backend', 'lattigo')
        )
    
    def _create_scheme_from_orion(self, orion_scheme: Any) -> OrionSchemeParameters:
        """Create scheme parameters from Orion scheme object."""
        return OrionSchemeParameters(
            logN=getattr(orion_scheme, 'logN', [13]),
            logQ=getattr(orion_scheme, 'logQ', [55, 45, 45, 55]),
            logP=getattr(orion_scheme, 'logP', [55]),
            logScale=getattr(orion_scheme, 'logScale', 45),
            slots=getattr(orion_scheme, 'slots', 2**12),
            ring_degree=getattr(orion_scheme, 'ring_degree', 2**13),
            backend=getattr(orion_scheme, 'backend', 'lattigo')
        )
    
    def _create_default_scheme(self) -> OrionSchemeParameters:
        """Create default scheme parameters."""
        return OrionSchemeParameters(
            logN=[13],
            logQ=[55, 45, 45, 55],
            logP=[55],
            logScale=45,
            slots=2**12,
            ring_degree=2**13,
            backend='lattigo'
        )
    
    def translate_from_config(self, config_path: Path, operations: List[Any]) -> Any:
        """
        Convenience method to translate operations using config file.
        
        Args:
            config_path: Path to Orion configuration file
            operations: List of Orion operations
            
        Returns:
            HEIR MLIR module
        """
        from ...core.translator import GenericTranslator
        
        # Extract operations and parameters
        fhe_operations = self.extract_operations(operations)
        scheme_params = self.extract_scheme_parameters(config_path)
        
        # Create translator and translate
        translator = GenericTranslator()
        return translator.translate(fhe_operations, scheme_params)
    
    def create_mlp_operations(self) -> List[FHEOperation]:
        """Create sample MLP operations for testing."""
        import torch
        
        operations = [
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="linear1",
                level=3
            ),
            FHEOperation(
                op_type="add_plain",
                method_name="add_plain", 
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="bias1",
                level=3
            ),
            FHEOperation(
                op_type="rotate",
                method_name="rotate",
                args=[1],
                kwargs={"offset": 1},
                result_var="rot1",
                level=3
            ),
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="linear2",
                level=2
            ),
            FHEOperation(
                op_type="add",
                method_name="add",
                args=[],
                kwargs={},
                result_var="accumulate",
                level=2
            )
        ]
        
        return operations


def create_orion_frontend() -> OrionFrontend:
    """Factory function to create an Orion frontend."""
    return OrionFrontend()


def translate_orion_operations(operations: List[Any], 
                              config_path: Optional[Path] = None) -> Any:
    """
    Convenience function to translate Orion operations.
    
    Args:
        operations: List of Orion operations
        config_path: Optional path to configuration file
        
    Returns:
        HEIR MLIR module
    """
    frontend = OrionFrontend()
    
    # Extract FHE operations
    fhe_operations = frontend.extract_operations(operations)
    
    # Extract scheme parameters
    if config_path:
        scheme_params = frontend.extract_scheme_parameters(config_path)
    else:
        scheme_params = frontend._create_default_scheme()
    
    # Translate
    from ...core.translator import GenericTranslator
    translator = GenericTranslator()
    
    return translator.translate(fhe_operations, scheme_params)
"""
Orion frontend for the HEIR translator.

This module provides Orion-specific functionality for extracting operations
and scheme parameters from Orion FHE computations.
"""

from typing import List, Any, Dict, Optional
import yaml
from pathlib import Path

from ...core.translator import FHEOperation, FrontendInterface, SchemeParameters
from .scheme_params import OrionSchemeParameters
from .operation_extractor import OrionOperationExtractor


class OrionFrontend(FrontendInterface):
    """
    Frontend for translating Orion FHE operations to HEIR.
    
    This class implements the FrontendInterface for Orion,
    providing Orion-specific logic for operation extraction
    and parameter handling.
    """
    
    def __init__(self):
        self.operation_extractor = OrionOperationExtractor()
    
    def extract_operations(self, source: Any) -> List[FHEOperation]:
        """
        Extract FHE operations from Orion source.
        
        Args:
            source: Can be a list of operations, trace data, or other Orion artifacts
            
        Returns:
            List of generic FHE operations
        """
        if isinstance(source, list):
            # Direct list of operations
            return self.operation_extractor.convert_operations(source)
        elif hasattr(source, 'operations'):
            # Object with operations attribute
            return self.operation_extractor.convert_operations(source.operations)
        else:
            # Try to extract from other formats
            return self.operation_extractor.extract_from_source(source)
    
    def extract_scheme_parameters(self, source: Any) -> SchemeParameters:
        """
        Extract scheme parameters from Orion source.
        
        Args:
            source: Orion configuration, scheme object, or config file path
            
        Returns:
            Scheme parameters object
        """
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
            logN=ckks_params.get('LogN', [13]),
            logQ=ckks_params.get('LogQ', [55, 45, 45, 55]),
            logP=ckks_params.get('LogP', [55]),
            logScale=ckks_params.get('LogScale', 45),
            slots=ckks_params.get('slots', 2**12),
            ring_degree=2**ckks_params.get('LogN', [13])[0],
            backend=orion_params.get('backend', 'lattigo')
        )
    
    def _create_scheme_from_orion(self, orion_scheme: Any) -> OrionSchemeParameters:
        """Create scheme parameters from Orion scheme object."""
        return OrionSchemeParameters(
            logN=getattr(orion_scheme, 'logN', [13]),
            logQ=getattr(orion_scheme, 'logQ', [55, 45, 45, 55]),
            logP=getattr(orion_scheme, 'logP', [55]),
            logScale=getattr(orion_scheme, 'logScale', 45),
            slots=getattr(orion_scheme, 'slots', 2**12),
            ring_degree=getattr(orion_scheme, 'ring_degree', 2**13),
            backend=getattr(orion_scheme, 'backend', 'lattigo')
        )
    
    def _create_default_scheme(self) -> OrionSchemeParameters:
        """Create default scheme parameters."""
        return OrionSchemeParameters(
            logN=[13],
            logQ=[55, 45, 45, 55],
            logP=[55],
            logScale=45,
            slots=2**12,
            ring_degree=2**13,
            backend='lattigo'
        )
    
    def translate_from_config(self, config_path: Path, operations: List[Any]) -> Any:
        """
        Convenience method to translate operations using config file.
        
        Args:
            config_path: Path to Orion configuration file
            operations: List of Orion operations
            
        Returns:
            HEIR MLIR module
        """
        from ...core.translator import GenericTranslator
        
        # Extract operations and parameters
        fhe_operations = self.extract_operations(operations)
        scheme_params = self.extract_scheme_parameters(config_path)
        
        # Create translator and translate
        translator = GenericTranslator()
        return translator.translate(fhe_operations, scheme_params)
    
    def create_mlp_operations(self) -> List[FHEOperation]:
        """Create sample MLP operations for testing."""
        import torch
        
        operations = [
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="linear1",
                level=3
            ),
            FHEOperation(
                op_type="add_plain",
                method_name="add_plain", 
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="bias1",
                level=3
            ),
            FHEOperation(
                op_type="rotate",
                method_name="rotate",
                args=[1],
                kwargs={"offset": 1},
                result_var="rot1",
                level=3
            ),
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="linear2",
                level=2
            ),
            FHEOperation(
                op_type="add",
                method_name="add",
                args=[],
                kwargs={},
                result_var="accumulate",
                level=2
            )
        ]
        
        return operations


def create_orion_frontend() -> OrionFrontend:
    """Factory function to create an Orion frontend."""
    return OrionFrontend()


def translate_orion_operations(operations: List[Any], 
                              config_path: Optional[Path] = None) -> Any:
    """
    Convenience function to translate Orion operations.
    
    Args:
        operations: List of Orion operations
        config_path: Optional path to configuration file
        
    Returns:
        HEIR MLIR module
    """
    frontend = OrionFrontend()
    
    # Extract FHE operations
    fhe_operations = frontend.extract_operations(operations)
    
    # Extract scheme parameters
    if config_path:
        scheme_params = frontend.extract_scheme_parameters(config_path)
    else:
        scheme_params = frontend._create_default_scheme()
    
    # Translate
    from ...core.translator import GenericTranslator
    translator = GenericTranslator()
    
    return translator.translate(fhe_operations, scheme_params)
