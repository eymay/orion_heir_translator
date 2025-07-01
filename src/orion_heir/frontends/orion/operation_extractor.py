"""
Orion operation extractor for converting Orion operations to generic FHE operations.

This module handles the extraction and conversion of operations from Orion
traces, method calls, or other Orion artifacts into the generic FHEOperation format.
"""

from typing import List, Any, Dict, Optional
import torch
from dataclasses import dataclass

from ...core.types import FHEOperation


@dataclass
class OrionOperationData:
    """Data class for Orion-specific operation information."""
    op_type: str
    method_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    result_var: Optional[str] = None
    level: Optional[int] = None
    orion_metadata: Optional[Dict[str, Any]] = None


class OrionOperationExtractor:
    """
    Extractor for converting Orion operations to generic FHE operations.
    
    This class handles the specifics of Orion operation formats and
    converts them to the generic FHEOperation format used by the translator.
    """
    
    def __init__(self):
        self.operation_mapping = self._create_operation_mapping()
    
    def _create_operation_mapping(self) -> Dict[str, str]:
        """Create mapping from Orion operation names to generic names."""
        return {
            # Orion method names -> Generic operation types
            'add': 'add',
            'add_plain': 'add_plain',
            'sub': 'sub',
            'sub_plain': 'sub_plain',
            'mul': 'mul',
            'mul_plain': 'mul_plain',
            'rotate': 'rotate',
            'rot': 'rotate',
            'linear_transform': 'linear_transform',
            'linear': 'linear_transform',
            'matmul': 'linear_transform',
            'encode': 'encode',
            'encrypt': 'encrypt',
            'decrypt': 'decrypt',
            'rescale': 'rescale',
            'relinearize': 'relinearize',
            'bootstrap': 'bootstrap'
        }
    
    def convert_operations(self, orion_operations: List[Any]) -> List[FHEOperation]:
        """
        Convert a list of Orion operations to generic FHE operations.
        
        Args:
            orion_operations: List of Orion operation objects or data
            
        Returns:
            List of generic FHE operations
        """
        fhe_operations = []
        
        for i, op in enumerate(orion_operations):
            try:
                fhe_op = self._convert_single_operation(op, i)
                if fhe_op:
                    fhe_operations.append(fhe_op)
            except Exception as e:
                print(f"⚠️ Error converting operation {i}: {e}")
                continue
        
        print(f"✅ Converted {len(fhe_operations)} Orion operations")
        return fhe_operations
    
    def _convert_single_operation(self, orion_op: Any, index: int) -> Optional[FHEOperation]:
        """Convert a single Orion operation to FHEOperation."""
        
        if isinstance(orion_op, dict):
            return self._convert_from_dict(orion_op, index)
        elif hasattr(orion_op, '__dict__'):
            return self._convert_from_object(orion_op, index)
        elif isinstance(orion_op, OrionOperationData):
            return self._convert_from_orion_data(orion_op)
        else:
            # Try to extract from other formats
            return self._convert_from_unknown(orion_op, index)
    
    def _convert_from_dict(self, op_dict: Dict[str, Any], index: int) -> FHEOperation:
        """Convert operation from dictionary format."""
        op_type = op_dict.get('op_type', op_dict.get('method_name', f'unknown_{index}'))
        method_name = op_dict.get('method_name', op_type)
        
        # Map Orion operation name to generic name
        generic_op_type = self.operation_mapping.get(op_type, op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=method_name,
            args=op_dict.get('args', []),
            kwargs=op_dict.get('kwargs', {}),
            result_var=op_dict.get('result_var'),
            level=op_dict.get('level'),
            metadata={'orion_data': op_dict, 'source': 'dict'}
        )
    
    def _convert_from_object(self, orion_op: Any, index: int) -> FHEOperation:
        """Convert operation from Orion object with attributes."""
        op_type = getattr(orion_op, 'op_type', getattr(orion_op, 'method_name', f'unknown_{index}'))
        method_name = getattr(orion_op, 'method_name', op_type)
        
        # Map to generic operation type
        generic_op_type = self.operation_mapping.get(op_type, op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=method_name,
            args=getattr(orion_op, 'args', []),
            kwargs=getattr(orion_op, 'kwargs', {}),
            result_var=getattr(orion_op, 'result_var', None),
            level=getattr(orion_op, 'level', None),
            metadata={'orion_object': orion_op, 'source': 'object'}
        )
    
    def _convert_from_orion_data(self, orion_data: OrionOperationData) -> FHEOperation:
        """Convert from OrionOperationData object."""
        generic_op_type = self.operation_mapping.get(orion_data.op_type, orion_data.op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=orion_data.method_name,
            args=orion_data.args,
            kwargs=orion_data.kwargs,
            result_var=orion_data.result_var,
            level=orion_data.level,
            metadata=orion_data.orion_metadata or {}
        )
    
    def _convert_from_unknown(self, orion_op: Any, index: int) -> Optional[FHEOperation]:
        """Try to convert from unknown format."""
        # Try to extract information from string representation
        if isinstance(orion_op, str):
            return FHEOperation(
                op_type=orion_op,
                method_name=orion_op,
                args=[],
                kwargs={},
                result_var=f"result_{index}",
                metadata={'source': 'string'}
            )
        
        print(f"⚠️ Unknown operation format for operation {index}: {type(orion_op)}")
        return None
    
    def extract_from_source(self, source: Any) -> List[FHEOperation]:
        """
        Extract operations from various Orion source formats.
        
        Args:
            source: Orion computation trace, log, or other source
            
        Returns:
            List of extracted FHE operations
        """
        if isinstance(source, str):
            return self._extract_from_string(source)
        elif isinstance(source, dict):
            return self._extract_from_trace_dict(source)
        elif hasattr(source, 'operations'):
            return self.convert_operations(source.operations)
        else:
            print(f"⚠️ Unknown source format: {type(source)}")
            return []
    
    def _extract_from_string(self, source: str) -> List[FHEOperation]:
        """Extract operations from string representation."""
        # Simple parser for string-based operation descriptions
        operations = []
        lines = source.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse simple format: "operation_name(args)"
            parts = line.split('(')
            if len(parts) >= 2:
                op_name = parts[0].strip()
                args_str = parts[1].rstrip(')')
                
                # Simple argument parsing
                args = []
                if args_str:
                    args = [arg.strip() for arg in args_str.split(',')]
                
                generic_op_type = self.operation_mapping.get(op_name, op_name)
                
                operations.append(FHEOperation(
                    op_type=generic_op_type,
                    method_name=op_name,
                    args=args,
                    kwargs={},
                    result_var=f"result_{i}",
                    metadata={'source': 'string_parse'}
                ))
        
        return operations
    
    def _extract_from_trace_dict(self, trace: Dict[str, Any]) -> List[FHEOperation]:
        """Extract operations from Orion trace dictionary."""
        operations = []
        
        # Look for operations in various trace formats
        if 'operations' in trace:
            return self.convert_operations(trace['operations'])
        elif 'computation_trace' in trace:
            return self.convert_operations(trace['computation_trace'])
        elif 'steps' in trace:
            for step in trace['steps']:
                if isinstance(step, dict) and 'operation' in step:
                    op = self._convert_from_dict(step['operation'], len(operations))
                    operations.append(op)
        
        return operations
    
    def create_sample_mlp_operations(self) -> List[FHEOperation]:
        """Create sample MLP operations for testing."""
        operations = [
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="weight1",
                level=3,
                metadata={'layer': 'linear1'}
            ),
            FHEOperation(
                op_type="add_plain",
                method_name="add_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="bias1",
                level=3,
                metadata={'layer': 'linear1'}
            ),
            FHEOperation(
                op_type="rotate",
                method_name="rotate",
                args=[1],
                kwargs={"offset": 1},
                result_var="shift1",
                level=3,
                metadata={'purpose': 'alignment'}
            ),
            FHEOperation(
                op_type="linear_transform",
                method_name="linear_transform",
                args=[torch.randn(16, 8)],
                kwargs={},
                result_var="weight2",
                level=2,
                metadata={'layer': 'linear2'}
            ),
            FHEOperation(
                op_type="add",
                method_name="add",
                args=[],
                kwargs={},
                result_var="accumulate",
                level=2,
                metadata={'purpose': 'accumulation'}
            )
        ]
        
        return operations


def create_orion_extractor() -> OrionOperationExtractor:
    """Factory function to create an Orion operation extractor."""
    return OrionOperationExtractor()


def extract_orion_operations(source: Any) -> List[FHEOperation]:
    """
    Convenience function to extract operations from Orion source.
    
    Args:
        source: Orion operations in any supported format
        
    Returns:
        List of generic FHE operations
    """
    extractor = OrionOperationExtractor()
    return extractor.extract_from_source(source)
"""
Orion operation extractor for converting Orion operations to generic FHE operations.

This module handles the extraction and conversion of operations from Orion
traces, method calls, or other Orion artifacts into the generic FHEOperation format.
"""

from typing import List, Any, Dict, Optional
import torch
from dataclasses import dataclass

from ...core.translator import FHEOperation


@dataclass
class OrionOperationData:
    """Data class for Orion-specific operation information."""
    op_type: str
    method_name: str
    args: List[Any]
    kwargs: Dict[str, Any]
    result_var: Optional[str] = None
    level: Optional[int] = None
    orion_metadata: Optional[Dict[str, Any]] = None


class OrionOperationExtractor:
    """
    Extractor for converting Orion operations to generic FHE operations.
    
    This class handles the specifics of Orion operation formats and
    converts them to the generic FHEOperation format used by the translator.
    """
    
    def __init__(self):
        self.operation_mapping = self._create_operation_mapping()
    
    def _create_operation_mapping(self) -> Dict[str, str]:
        """Create mapping from Orion operation names to generic names."""
        return {
            # Orion method names -> Generic operation types
            'add': 'add',
            'add_plain': 'add_plain',
            'sub': 'sub',
            'sub_plain': 'sub_plain',
            'mul': 'mul',
            'mul_plain': 'mul_plain',
            'rotate': 'rotate',
            'rot': 'rotate',
            'linear_transform': 'linear_transform',
            'linear': 'linear_transform',
            'matmul': 'linear_transform',
            'encode': 'encode',
            'encrypt': 'encrypt',
            'decrypt': 'decrypt',
            'rescale': 'rescale',
            'relinearize': 'relinearize',
            'bootstrap': 'bootstrap'
        }
    
    def convert_operations(self, orion_operations: List[Any]) -> List[FHEOperation]:
        """
        Convert a list of Orion operations to generic FHE operations.
        
        Args:
            orion_operations: List of Orion operation objects or data
            
        Returns:
            List of generic FHE operations
        """
        fhe_operations = []
        
        for i, op in enumerate(orion_operations):
            try:
                fhe_op = self._convert_single_operation(op, i)
                if fhe_op:
                    fhe_operations.append(fhe_op)
            except Exception as e:
                print(f"⚠️ Error converting operation {i}: {e}")
                continue
        
        print(f"✅ Converted {len(fhe_operations)} Orion operations")
        return fhe_operations
    
    def _convert_single_operation(self, orion_op: Any, index: int) -> Optional[FHEOperation]:
        """Convert a single Orion operation to FHEOperation."""
        
        if isinstance(orion_op, dict):
            return self._convert_from_dict(orion_op, index)
        elif hasattr(orion_op, '__dict__'):
            return self._convert_from_object(orion_op, index)
        elif isinstance(orion_op, OrionOperationData):
            return self._convert_from_orion_data(orion_op)
        else:
            # Try to extract from other formats
            return self._convert_from_unknown(orion_op, index)
    
    def _convert_from_dict(self, op_dict: Dict[str, Any], index: int) -> FHEOperation:
        """Convert operation from dictionary format."""
        op_type = op_dict.get('op_type', op_dict.get('method_name', f'unknown_{index}'))
        method_name = op_dict.get('method_name', op_type)
        
        # Map Orion operation name to generic name
        generic_op_type = self.operation_mapping.get(op_type, op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=method_name,
            args=op_dict.get('args', []),
            kwargs=op_dict.get('kwargs', {}),
            result_var=op_dict.get('result_var'),
            level=op_dict.get('level'),
            metadata={'orion_data': op_dict, 'source': 'dict'}
        )
    
    def _convert_from_object(self, orion_op: Any, index: int) -> FHEOperation:
        """Convert operation from Orion object with attributes."""
        op_type = getattr(orion_op, 'op_type', getattr(orion_op, 'method_name', f'unknown_{index}'))
        method_name = getattr(orion_op, 'method_name', op_type)
        
        # Map to generic operation type
        generic_op_type = self.operation_mapping.get(op_type, op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=method_name,
            args=getattr(orion_op, 'args', []),
            kwargs=getattr(orion_op, 'kwargs', {}),
            result_var=getattr(orion_op, 'result_var', None),
            level=getattr(orion_op, 'level', None),
            metadata={'orion_object': orion_op, 'source': 'object'}
        )
    
    def _convert_from_orion_data(self, orion_data: OrionOperationData) -> FHEOperation:
        """Convert from OrionOperationData object."""
        generic_op_type = self.operation_mapping.get(orion_data.op_type, orion_data.op_type)
        
        return FHEOperation(
            op_type=generic_op_type,
            method_name=orion_data.method_name,
            args=orion_data.args,
            kwargs=orion_data.kwargs,
            result_var=orion_data.result_var,
            level=orion_data.level,
            metadata=orion_data.orion_metadata or {}
        )
    
    def _convert_from_unknown(self, orion_op: Any, index: int) -> Optional[FHEOperation]:
        """Try to convert from unknown format."""
        # Try to extract information from string representation
        if isinstance(orion_op, str):
            return FHEOperation(
                op_type=orion_op,
                method_name=orion_op,
                args=[],
                kwargs={},
                result_var=f"result_{index}",
                metadata={'source': 'string'}
            )
        
        print(f"⚠️ Unknown operation format for operation {index}: {type(orion_op)}")
        return None
    
    def extract_from_source(self, source: Any) -> List[FHEOperation]:
        """
        Extract operations from various Orion source formats.
        
        Args:
            source: Orion computation trace, log, or other source
            
        Returns:
            List of extracted FHE operations
        """
        if isinstance(source, str):
            return self._extract_from_string(source)
        elif isinstance(source, dict):
            return self._extract_from_trace_dict(source)
        elif hasattr(source, 'operations'):
            return self.convert_operations(source.operations)
        else:
            print(f"⚠️ Unknown source format: {type(source)}")
            return []
    
    def _extract_from_string(self, source: str) -> List[FHEOperation]:
        """Extract operations from string representation."""
        # Simple parser for string-based operation descriptions
        operations = []
        lines = source.strip().split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse simple format: "operation_name(args)"
            parts = line.split('(')
            if len(parts) >= 2:
                op_name = parts[0].strip()
                args_str = parts[1].rstrip(')')
                
                # Simple argument parsing
                args = []
                if args_str:
                    args = [arg.strip() for arg in args_str.split(',')]
                
                generic_op_type = self.operation_mapping.get(op_name, op_name)
                
                operations.append(FHEOperation(
                    op_type=generic_op_type,
                    method_name=op_name,
                    args=args,
                    kwargs={},
                    result_var=f"result_{i}",
                    metadata={'source': 'string_parse'}
                ))
        
        return operations
    
    def _extract_from_trace_dict(self, trace: Dict[str, Any]) -> List[FHEOperation]:
        """Extract operations from Orion trace dictionary."""
        operations = []
        
        # Look for operations in various trace formats
        if 'operations' in trace:
            return self.convert_operations(trace['operations'])
        elif 'computation_trace' in trace:
            return self.convert_operations(trace['computation_trace'])
        elif 'steps' in trace:
            for step in trace['steps']:
                if isinstance(step, dict) and 'operation' in step:
                    op = self._convert_from_dict(step['operation'], len(operations))
                    operations.append(op)
        
        return operations
    
    def create_sample_mlp_operations(self) -> List[FHEOperation]:
        """Create sample MLP operations for testing."""
        operations = [
            FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="weight1",
                level=3,
                metadata={'layer': 'linear1'}
            ),
            FHEOperation(
                op_type="add_plain",
                method_name="add_plain",
                args=[torch.randn(1, 16)],
                kwargs={},
                result_var="bias1",
                level=3,
                metadata={'layer': 'linear1'}
            ),
            FHEOperation(
                op_type="rotate",
                method_name="rotate",
                args=[1],
                kwargs={"offset": 1},
                result_var="shift1",
                level=3,
                metadata={'purpose': 'alignment'}
            ),
            FHEOperation(
                op_type="linear_transform",
                method_name="linear_transform",
                args=[torch.randn(16, 8)],
                kwargs={},
                result_var="weight2",
                level=2,
                metadata={'layer': 'linear2'}
            ),
            FHEOperation(
                op_type="add",
                method_name="add",
                args=[],
                kwargs={},
                result_var="accumulate",
                level=2,
                metadata={'purpose': 'accumulation'}
            )
        ]
        
        return operations


def create_orion_extractor() -> OrionOperationExtractor:
    """Factory function to create an Orion operation extractor."""
    return OrionOperationExtractor()


def extract_orion_operations(source: Any) -> List[FHEOperation]:
    """
    Convenience function to extract operations from Orion source.
    
    Args:
        source: Orion operations in any supported format
        
    Returns:
        List of generic FHE operations
    """
    extractor = OrionOperationExtractor()
    return extractor.extract_from_source(source)
