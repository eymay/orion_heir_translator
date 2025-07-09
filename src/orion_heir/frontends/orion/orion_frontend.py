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
    
    def extract_operations(self, model: Any) -> List[FHEOperation]:
        """
        Extract FHE operations from a compiled Orion model.
        
        This method extracts the actual operations that Orion performs during
        FHE inference, including proper handling of fused operations.
        """
        operations = []
        
        # Track expected vs found operations for warnings
        expected_layers = []
        found_layers = []
        
        # Get all layers from the model
        all_layers = list(model.named_modules())
        
        # Identify what we expect to find
        for name, layer in all_layers:
            if name:  # Skip empty names (root module)
                layer_type = layer.__class__.__name__
                expected_layers.append(f"{name}({layer_type})")
        
        print(f"📋 Expected layers: {expected_layers}")
        
        # Process each layer
        for name, layer in all_layers:
            if not name:  # Skip root module
                continue
                
            layer_type = layer.__class__.__name__
            
            # Check if this is a layer that should produce FHE operations
            if self._should_extract_layer(layer):
                print(f"🔍 Processing layer: {name} ({layer_type})")
                
                # Get operations based on layer type
                if layer_type == 'Linear':
                    layer_ops = self._get_linear_operations(layer, name)
                    operations.extend(layer_ops)
                    found_layers.append(f"{name}(Linear)")
                    
                elif layer_type == 'Quad':
                    layer_ops = self._get_quad_operations(layer, name)
                    operations.extend(layer_ops)
                    found_layers.append(f"{name}(Quad)")
                    
                elif layer_type == 'BatchNorm1d':
                    # Check if BatchNorm is fused into adjacent Linear layer
                    if self._is_batchnorm_fused(layer):
                        print(f"    ℹ️  BatchNorm {name} is fused into adjacent Linear layer")
                        found_layers.append(f"{name}(BatchNorm1d-fused)")
                    else:
                        layer_ops = self._get_batchnorm_operations(layer, name)
                        operations.extend(layer_ops)
                        found_layers.append(f"{name}(BatchNorm1d)")
                        
                elif layer_type == 'Flatten':
                    # Flatten is plaintext operation, handled by Linear diagonals
                    print(f"    ℹ️  Flatten {name} is plaintext operation (handled by Linear diagonals)")
                    found_layers.append(f"{name}(Flatten-plaintext)")
                    
                else:
                    print(f"    ⚠️  Unknown layer type: {layer_type}")
            else:
                # Layer doesn't produce FHE operations
                if layer_type in ['Flatten', 'BatchNorm1d']:
                    print(f"    ℹ️  {name}({layer_type}) - no direct FHE operations")
                else:
                    print(f"    ⚠️  {name}({layer_type}) - no operations extracted")
        
        # Report what we found vs expected
        print(f"\n📊 Extraction Summary:")
        print(f"   Expected layers: {len(expected_layers)}")
        print(f"   Processed layers: {len(found_layers)}")
        print(f"   Total operations: {len(operations)}")
        
        # Check for missing operations
        missing_activations = []
        for name, layer in all_layers:
            if layer.__class__.__name__ == 'Quad' and f"{name}(Quad)" not in found_layers:
                missing_activations.append(name)
        
        if missing_activations:
            print(f"   ⚠️  Missing Quad activations: {missing_activations}")
        
        return operations

    def _should_extract_layer(self, layer: Any) -> bool:
        """
        Check if a layer should produce FHE operations.
        
        This expanded version handles more layer types beyond just Linear.
        """
        layer_type = layer.__class__.__name__
        
        # Linear layers always produce operations
        if layer_type == 'Linear':
            return hasattr(layer, 'diagonals') or hasattr(layer, 'transform_ids')
        
        # Quad activations produce operations
        if layer_type == 'Quad':
            return hasattr(layer, 'level')  # Compiled Quad layers have level
        
        # BatchNorm may produce operations if not fused
        if layer_type == 'BatchNorm1d':
            return not self._is_batchnorm_fused(layer)
        
        # Flatten is plaintext, no FHE operations needed
        if layer_type == 'Flatten':
            return False
        
        return False
    
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
        Get operations for a Linear layer with proper diagonal data extraction.
        """
        operations = []
        level = getattr(layer, 'level', 1)
        
        # Extract comprehensive Orion metadata
        orion_metadata = {
            'operation': 'orion_linear_transform',
            'layer': layer_name,
            'layer_type': 'Linear',
            'orion_level': level,
        }
        
        # Get transform information
        if hasattr(layer, 'transform_ids') and layer.transform_ids:
            orion_metadata['transform_blocks'] = len(layer.transform_ids)
            orion_metadata['transform_ids'] = list(layer.transform_ids.keys())
        
        # Get diagonal information
        if hasattr(layer, 'diagonals') and layer.diagonals:
            total_diagonals = sum(len(diags) for diags in layer.diagonals.values())
            orion_metadata['diagonal_count'] = total_diagonals
            orion_metadata['diagonal_blocks'] = list(layer.diagonals.keys())
            
            # Get actual diagonal indices from first block
            first_block = next(iter(layer.diagonals.values()))
            if first_block:
                orion_metadata['diagonal_indices'] = list(first_block.keys())
        
        # Get matrix shape and other metadata (same as before)
        if hasattr(layer, 'weight'):
            orion_metadata['matrix_shape'] = list(layer.weight.shape)
        
        if hasattr(layer, 'bsgs_ratio'):
            orion_metadata['bsgs_ratio'] = layer.bsgs_ratio
        
        if hasattr(layer, 'scheme') and hasattr(layer.scheme, 'params'):
            embedding_method = getattr(layer.scheme.params, 'embedding_method', 'hybrid')
            orion_metadata['embedding_method'] = embedding_method
            
            if hasattr(layer.scheme.params, 'get_slots'):
                orion_metadata['slots'] = layer.scheme.params.get_slots()
        
        if hasattr(layer, 'output_rotations'):
            orion_metadata['output_rotations'] = layer.output_rotations
        
        # NEW: Extract the ACTUAL diagonal data from Orion
        diagonal_data = None
        
        print(f"    🔍 Extracting diagonal data from Orion layer {layer_name}...")
        
        if hasattr(layer, 'diagonals') and layer.diagonals:
            try:
                # Orion stores diagonals as: {(block_row, block_col): {diag_idx: diag_data}}
                diagonal_data = self.extract_orion_diagonals(layer)
                if diagonal_data is not None:
                    orion_metadata['has_diagonal_data'] = True
                    print(f"    ✅ Extracted diagonal data: {diagonal_data.shape}")
                else:
                    print(f"    ⚠️  Could not extract diagonal data from layer")
            except Exception as e:
                print(f"    ❌ Error extracting diagonal data: {e}")
                import traceback
                traceback.print_exc()
        
        # If no diagonal data, try to get it from the weight matrix directly
        if diagonal_data is None and hasattr(layer, 'weight'):
            try:
                print(f"    🔄 Fallback: Creating diagonals from weight matrix...")
                diagonal_data = create_diagonals_from_weight_matrix(layer)
                if diagonal_data is not None:
                    orion_metadata['has_diagonal_data'] = True
                    orion_metadata['diagonal_source'] = 'weight_matrix'
                    print(f"    ✅ Created diagonal data from weights: {diagonal_data.shape}")
            except Exception as e:
                print(f"    ❌ Error creating diagonals from weight matrix: {e}")
        
        # Create the linear transform operation with diagonal data
        linear_transform_args = []
        if diagonal_data is not None:
            linear_transform_args.append(diagonal_data)
        
        print(f"    📊 Final metadata: {orion_metadata}")
        
        operations.append(FHEOperation(
            op_type="linear_transform",
            method_name="linear_transform",
            args=linear_transform_args,  # Include actual diagonal data
            kwargs={},
            result_var=f"{layer_name}_linear",
            level=level,
            metadata=orion_metadata
        ))
        
        if hasattr(layer, 'output_rotations') and layer.output_rotations > 0:
            current_result = f"{layer_name}_linear"
            slots = orion_metadata.get('slots', 4096)
            
            for i in range(1, layer.output_rotations + 1):  # Note: starts from 1, not 0
                # Calculate rotation offset: slots // (2**i)
                rotation_offset = slots // (2**i)
                
                # Rotation - always from the original linear transform result
                operations.append(FHEOperation(
                    op_type="rotate",
                    method_name="rotate",
                    args=[rotation_offset],
                    kwargs={"offset": rotation_offset},
                    result_var=f"{layer_name}_rot_{i}",
                    level=level,
                    metadata={'operation': 'output_rotation', 'layer': layer_name}
                ))
                
                # Accumulation - add this rotation to the current result
                operations.append(FHEOperation(
                    op_type="add",
                    method_name="add",
                    args=[f"@{current_result}", f"@{layer_name}_rot_{i}"],
                    kwargs={},
                    result_var=f"{layer_name}_acc_{i}",
                    level=level,
                    metadata={'operation': 'rotation_accumulation', 'layer': layer_name}
                ))
                
                # Update current result for next iteration
                current_result = f"{layer_name}_acc_{i}"
            
            # Update the final result variable name for bias addition
            final_result = current_result
        else:
            final_result = f"{layer_name}_linear"

        # Bias addition (using the final accumulated result)
        if hasattr(layer, 'bias') and layer.bias is not None:
            operations.append(FHEOperation(
                op_type="encode",
                method_name="encode",
                args=[layer.bias],
                kwargs={},
                result_var=f"{layer_name}_bias_encoded",
                level=level,
                metadata={'operation': 'lwe_encode', 'layer': layer_name}
            ))
            
            operations.append(FHEOperation(
                op_type="add_plain",
                method_name="add_plain",
                args=[f"@{final_result}", f"@{layer_name}_bias_encoded"],
                kwargs={},
                result_var=f"{layer_name}_result",
                level=level,
                metadata={'operation': 'bias_addition', 'layer': layer_name}
            ))
        return operations

    def extract_orion_diagonals(self, layer: Any) -> Optional[torch.Tensor]:
        """Extract diagonal data from Orion layer with comprehensive checking."""
        import torch
        import numpy as np
        from typing import Optional, Any
        
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
        
        for diag_idx in diagonal_indices[:5]:  # Check first 5 diagonals
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
            
            # Convert to tensor
            if isinstance(diag_data, torch.Tensor):
                diagonal_list.append(diag_data.flatten())
            elif isinstance(diag_data, (list, np.ndarray)):
                diagonal_list.append(torch.tensor(diag_data, dtype=torch.float32).flatten())
            else:
                print(f"         ❌ Unknown diagonal type: {type(diag_data)}")
                continue
        
        if not diagonal_list:
            print(f"      ❌ No valid diagonal data found")
            return None
        
        try:
            result = torch.stack(diagonal_list)
            print(f"      ✅ Extracted diagonal tensor: {result.shape}")
            return result
        except Exception as e:
            print(f"      ❌ Error stacking diagonals: {e}")
            return None

    
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

    def _get_quad_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for a Quad activation layer.
        
        Quad activation: f(x) = x^2
        This requires a self-multiplication operation in FHE.
        """
        operations = []
        level = getattr(layer, 'level', 1)
        
        print(f"    🔢 Quad activation: x^2 at level {level}")
        
        # Quadratic activation: x * x
        operations.append(FHEOperation(
            op_type="quad",
            method_name="quad",
            args=[],  # Self-multiplication, no additional args needed
            kwargs={},
            result_var=f"{layer_name}_result",
            level=level,
            metadata={
                'operation': 'quadratic_activation',
                'layer': layer_name,
                'layer_type': 'Quad',
                'polynomial_degree': 2,
                'level_consumed': True  # This operation consumes a level
            }
        ))
        
        return operations

    def _get_batchnorm_operations(self, layer: Any, layer_name: str) -> List[FHEOperation]:
        """
        Get operations for a standalone BatchNorm layer.
        
        BatchNorm: y = (x - mean) / std * weight + bias
        In FHE: y = x * (weight/std) + (bias - mean*weight/std)
        """
        operations = []
        level = getattr(layer, 'level', 1)
        
        print(f"    📊 Standalone BatchNorm at level {level}")
        
        # Get BatchNorm parameters
        weight = layer.weight.data if hasattr(layer, 'weight') and layer.weight is not None else None
        bias = layer.bias.data if hasattr(layer, 'bias') and layer.bias is not None else None
        running_mean = layer.running_mean.data if hasattr(layer, 'running_mean') else None
        running_var = layer.running_var.data if hasattr(layer, 'running_var') else None
        eps = getattr(layer, 'eps', 1e-5)
        
        if weight is not None and running_var is not None:
            # Compute fused parameters
            import torch
            std = torch.sqrt(running_var + eps)
            fused_weight = weight / std
            fused_bias = bias - running_mean * fused_weight if bias is not None and running_mean is not None else None
            
            # Step 1: Multiply by (weight/std)
            operations.append(FHEOperation(
                op_type="encode",
                method_name="rlwe_encode",
                args=[fused_weight],
                kwargs={},
                result_var=f"{layer_name}_weight_encoded",
                level=level,
                metadata={
                    'operation': 'lwe_encode',
                    'layer': layer_name,
                    'layer_type': 'BatchNorm1d'
                }
            ))
            
            operations.append(FHEOperation(
                op_type="mul_plain",
                method_name="mul_plain",
                args=[f"@{layer_name}_weight_encoded"],
                kwargs={},
                result_var=f"{layer_name}_normalized",
                level=level,
                metadata={
                    'operation': 'batchnorm_scale',
                    'layer': layer_name,
                    'layer_type': 'BatchNorm1d'
                }
            ))
            
            # Step 2: Add bias if present
            if fused_bias is not None:
                operations.append(FHEOperation(
                    op_type="encode",
                    method_name="rlwe_encode",
                    args=[fused_bias],
                    kwargs={},
                    result_var=f"{layer_name}_bias_encoded",
                    level=level,
                    metadata={
                        'operation': 'lwe_encode',
                        'layer': layer_name,
                        'layer_type': 'BatchNorm1d'
                    }
                ))
                
                operations.append(FHEOperation(
                    op_type="add_plain",
                    method_name="add_plain",
                    args=[f"@{layer_name}_bias_encoded"],
                    kwargs={},
                    result_var=f"{layer_name}_result",
                    level=level,
                    metadata={
                        'operation': 'batchnorm_bias',
                        'layer': layer_name,
                        'layer_type': 'BatchNorm1d'
                    }
                ))
        
        return operations

    def _is_batchnorm_fused(self, layer: Any) -> bool:
        """
        Check if a BatchNorm layer is fused into an adjacent Linear layer.
        
        In Orion, BatchNorm is often fused into the preceding Linear layer
        during compilation for efficiency.
        """
        # Simple heuristic: if BatchNorm doesn't have a level assigned,
        # it's likely fused into another layer
        return not hasattr(layer, 'level') or layer.level is None

    
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
