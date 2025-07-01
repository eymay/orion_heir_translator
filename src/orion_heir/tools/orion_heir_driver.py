#!/usr/bin/env python3
"""
Orion to HEIR translator command-line driver.

This tool provides a command-line interface for translating Orion FHE
operations to HEIR MLIR format.
"""

import click
import yaml
import json
from pathlib import Path
from typing import Optional, List, Any

from xdsl.printer import Printer

from ..core.translator import GenericTranslator
from ..frontends.orion.orion_frontend import OrionFrontend
from ..frontends.orion.operation_extractor import OrionOperationExtractor
from .common_utils import setup_logging, validate_file_path, create_output_directory


class OrionHeirDriver:
    """
    Main driver class for Orion to HEIR translation.
    
    This class orchestrates the translation process from Orion operations
    to HEIR MLIR, providing both programmatic and CLI interfaces.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.frontend = OrionFrontend()
        self.translator = GenericTranslator()
        
        if verbose:
            setup_logging('DEBUG')
        else:
            setup_logging('INFO')
    
    def translate_from_operations(self, 
                                 operations: List[Any],
                                 config_path: Optional[Path] = None,
                                 function_name: str = "fhe_computation") -> str:
        """
        Translate Orion operations to HEIR MLIR.
        
        Args:
            operations: List of Orion operations
            config_path: Optional path to configuration file
            function_name: Name for the generated function
            
        Returns:
            HEIR MLIR as string
        """
        print("🚀 Starting Orion to HEIR translation...")
        
        # Extract FHE operations
        fhe_operations = self.frontend.extract_operations(operations)
        print(f"📊 Extracted {len(fhe_operations)} operations")
        
        # Extract scheme parameters
        if config_path:
            scheme_params = self.frontend.extract_scheme_parameters(config_path)
            print(f"📋 Loaded scheme parameters from {config_path}")
        else:
            scheme_params = self.frontend._create_default_scheme()
            print("📋 Using default scheme parameters")
        
        # Translate to HEIR
        module = self.translator.translate(fhe_operations, scheme_params, function_name)
        
        # Convert to string
        printer = Printer()
        output = printer.print_to_string(module)
        
        print("✅ Translation completed successfully")
        return output
    
    def translate_from_file(self,
                           input_path: Path,
                           config_path: Optional[Path] = None,
                           function_name: str = "fhe_computation") -> str:
        """
        Translate operations from input file.
        
        Args:
            input_path: Path to input file (JSON, YAML, or text)
            config_path: Optional path to configuration file
            function_name: Name for the generated function
            
        Returns:
            HEIR MLIR as string
        """
        print(f"📂 Loading operations from {input_path}")
        
        # Load operations from file
        operations = self._load_operations_from_file(input_path)
        
        # Translate
        return self.translate_from_operations(operations, config_path, function_name)
    
    def translate_mlp_example(self, config_path: Optional[Path] = None) -> str:
        """
        Translate a sample MLP computation.
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            HEIR MLIR as string
        """
        print("🧠 Generating sample MLP operations...")
        
        # Create sample operations
        operations = self.frontend.create_mlp_operations()
        
        # Translate
        return self.translate_from_operations(operations, config_path, "mlp_inference")
    
    def _load_operations_from_file(self, file_path: Path) -> List[Any]:
        """Load operations from various file formats."""
        suffix = file_path.suffix.lower()
        
        with open(file_path, 'r') as f:
            if suffix == '.json':
                data = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                # Try to parse as text
                content = f.read()
                extractor = OrionOperationExtractor()
                return extractor._extract_from_string(content)
        
        # Extract operations from loaded data
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'operations' in data:
                return data['operations']
            else:
                return [data]
        else:
            raise ValueError(f"Unsupported data format in {file_path}")


@click.command()
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True, path_type=Path),
              help='Input file containing Orion operations (JSON, YAML, or text)')
@click.option('--output', '-o', 'output_path', type=click.Path(path_type=Path),
              help='Output file for HEIR MLIR (default: stdout)')
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True, path_type=Path),
              help='Configuration file with scheme parameters')
@click.option('--function-name', '-f', default='fhe_computation',
              help='Name for the generated function (default: fhe_computation)')
@click.option('--example', is_flag=True,
              help='Generate and translate sample MLP operations')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
@click.option('--validate', is_flag=True,
              help='Validate the generated MLIR (requires HEIR tools)')
def main(input_path: Optional[Path],
         output_path: Optional[Path],
         config_path: Optional[Path],
         function_name: str,
         example: bool,
         verbose: bool,
         validate: bool):
    """
    Translate Orion FHE operations to HEIR MLIR format.
    
    This tool converts Orion homomorphic encryption operations into HEIR
    MLIR representation for further optimization and code generation.
    
    Examples:
        # Translate operations from file
        orion-heir-translate -i operations.json -o output.mlir
        
        # Use configuration file
        orion-heir-translate -i ops.json -c config.yml -o output.mlir
        
        # Generate sample MLP example
        orion-heir-translate --example -o mlp_example.mlir
        
        # Output to stdout with verbose logging
        orion-heir-translate -i operations.json -v
    """
    try:
        # Create driver
        driver = OrionHeirDriver(verbose=verbose)
        
        # Determine translation source
        if example:
            if input_path:
                print("⚠️ Both --example and --input specified, using --example")
            mlir_output = driver.translate_mlp_example(config_path)
        elif input_path:
            mlir_output = driver.translate_from_file(input_path, config_path, function_name)
        else:
            # No input specified, show help
            ctx = click.get_current_context()
            click.echo(ctx.get_help())
            ctx.exit(1)
        
        # Validate if requested
        if validate:
            if not _validate_mlir(mlir_output):
                print("❌ MLIR validation failed")
                return 1
        
        # Output results
        if output_path:
            # Create output directory if needed
            create_output_directory(output_path.parent)
            
            # Write to file
            output_path.write_text(mlir_output)
            print(f"💾 Output written to {output_path}")
        else:
            # Output to stdout
            click.echo(mlir_output)
        
        print("🎉 Translation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


def _validate_mlir(mlir_content: str) -> bool:
    """
    Validate MLIR content using HEIR tools if available.
    
    Args:
        mlir_content: MLIR content to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Try to parse with xDSL
        from xdsl.parser import Parser
        from xdsl.context import Context
        from io import StringIO
        
        context = Context()
        # Load necessary dialects
        from ..dialects.ckks import CKKS
        from ..dialects.lwe import LWE
        from ..dialects.polynomial import Polynomial
        from ..dialects.mod_arith import ModArith
        from ..dialects.rns import RNS
        from ..dialects.mgmt import MGMT
        
        dialects = [CKKS, LWE, Polynomial, ModArith, RNS, MGMT]
        for dialect in dialects:
            context.load_dialect(dialect)
        
        parser = Parser(context, StringIO(mlir_content))
        module = parser.parse_module()
        
        print("✅ MLIR validation passed")
        return True
        
    except Exception as e:
        print(f"❌ MLIR validation failed: {e}")
        return False


# Additional CLI utilities
@click.group()
def cli():
    """Orion-HEIR translator toolkit."""
    pass


@cli.command()
@click.option('--output', '-o', type=click.Path(path_type=Path), default='sample_config.yml',
              help='Output path for sample configuration')
def create_config(output: Path):
    """Create a sample configuration file."""
    config = {
        'ckks_params': {
            'LogN': [13],
            'LogQ': [55, 45, 45, 55],
            'LogP': [55],
            'LogScale': 45,
            'H': 192,
            'RingType': 'standard'
        },
        'orion': {
            'margin': 2,
            'embedding_method': 'hybrid',
            'backend': 'lattigo',
            'fuse_modules': True,
            'debug': False,
            'diags_path': '',
            'keys_path': '',
            'io_mode': 'none'
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"📄 Sample configuration created at {output}")


@cli.command()
def info():
    """Display version and system information."""
    print("Orion-HEIR Translator")
    print("====================")
    print("Version: 0.1.0")
    print("Components:")
    print("  - xDSL HEIR dialects")
    print("  - Generic FHE translator")
    print("  - Orion frontend")
    print()
    
    # Check dependencies
    try:
        import orion
        print("✅ Orion FHE library available")
    except ImportError:
        print("⚠️ Orion FHE library not available")
    
    try:
        from xdsl import __version__
        print(f"✅ xDSL version: {__version__}")
    except:
        print("✅ xDSL available")


if __name__ == '__main__':
    cli()
