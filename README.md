# Orion-HEIR Translator

A standalone translator for converting Orion FHE operations to HEIR MLIR format.

## Overview

This project provides a clean, modular translator that converts homomorphic encryption operations from the Orion FHE library into HEIR (Homomorphic Encryption Intermediate Representation) MLIR format. The translator is designed with a generic core that can be extended to support multiple FHE libraries.

## Key Features

- **Standalone Project**: Self-contained with all necessary xDSL dialects
- **Generic Architecture**: Core translator can support multiple FHE frontends
- **Modular Design**: Clean separation between dialects, core logic, and frontends
- **Standard Dependencies**: Direct imports from upstream `orion` and `xdsl` packages
- **CLI Tool**: Command-line interface for easy usage
- **Extensible**: Easy to add new frontends and operations

## Project Structure

```
orion-heir-translator/
├── src/orion_heir/
│   ├── dialects/           # xDSL HEIR dialects (CKKS, LWE, etc.)
│   ├── core/              # Generic translator infrastructure  
│   ├── frontends/         # Frontend implementations (Orion, etc.)
│   │   └── orion/         # Orion-specific code
│   └── tools/             # CLI tools and drivers
├── tests/                 # Test suite
├── examples/              # Usage examples
└── docs/                 # Documentation
```

## Installation

### From PyPI (when published)
```bash
pip install orion-heir-translator
```

### From Source
```bash
git clone <repository-url>
cd orion-heir-translator
pip install -e .
```

### Dependencies
```bash
# Required
pip install xdsl torch pyyaml click

# Optional (for Orion frontend)
pip install orion-fhe  # If you have access to Orion
```

## Quick Start

### CLI Usage

```bash
# Generate and translate sample MLP operations
orion-heir-translate --example -o mlp_example.mlir

# Translate operations from file  
orion-heir-translate -i operations.json -o output.mlir

# Use configuration file
orion-heir-translate -i ops.json -c config.yml -o output.mlir

# Create sample configuration
orion-heir-translate create-config -o my_config.yml
```

### Python API

```python
from orion_heir import GenericTranslator, FHEOperation, OrionFrontend

# Create operations
operations = [
    FHEOperation(
        op_type="mul_plain",
        method_name="mul_plain", 
        args=[torch.tensor([[1.0, 2.0, 3.0, 4.0]])],
        kwargs={},
        result_var="weight"
    ),
    FHEOperation(
        op_type="rotate",
        method_name="rotate",
        args=[2],
        kwargs={"offset": 2},
        result_var="shift"
    )
]

# Create frontend and extract scheme parameters
frontend = OrionFrontend()
scheme_params = frontend._create_default_scheme()

# Translate to HEIR
translator = GenericTranslator() 
module = translator.translate(operations, scheme_params)

# Convert to MLIR string
from xdsl.printer import Printer
printer = Printer()
mlir_output = printer.print_to_string(module)
print(mlir_output)
```

## Architecture

### Core Components

1. **Generic Translator** (`core/translator.py`)
   - Frontend-agnostic translation engine
   - Handles operation sequencing and type management
   - Extensible operation registry

2. **Operation Registry** (`core/operation_registry.py`) 
   - Modular operation handlers
   - Easy to register new operation types
   - Factory pattern for similar operations

3. **Type Builder** (`core/type_builder.py`)
   - Constructs HEIR types from scheme parameters
   - Handles complex type system abstractions
   - Manages type inference

### HEIR Dialects

The project includes complete xDSL dialect implementations:

- **CKKS**: Homomorphic encryption operations (add, mul, rotate, etc.)
- **LWE**: Lattice-based cryptographic types and operations
- **Polynomial**: Polynomial ring attributes
- **ModArith**: Modular arithmetic types
- **RNS**: Residue number system types
- **MGMT**: Management operations for keys and parameters

### Orion Frontend

- **Operation Extractor**: Converts Orion operations to generic format
- **Scheme Parameters**: Integrates with Orion parameter system
- **Config Integration**: Loads parameters from YAML configuration files

## Extending the Translator

### Adding a New Frontend

```python
from orion_heir.core.translator import FrontendInterface, SchemeParameters

class MyFHEFrontend(FrontendInterface):
    def extract_operations(self, source):
        # Convert source operations to FHEOperation list
        pass
    
    def extract_scheme_parameters(self, source):
        # Extract scheme parameters
        pass
```

### Adding New Operations

```python
from orion_heir.core.operation_registry import BaseOperationHandler

class MyOperationHandler(BaseOperationHandler):
    def handle(self, operation, current_value, block, constants, type_builder):
        # Create MLIR operation
        pass

# Register the handler
translator.operation_registry.register_operation("my_op", MyOperationHandler())
```

## Configuration

### Sample Configuration File

```yaml
ckks_params:
  LogN: [13]              # Ring degree: 2^13 = 8192
  LogQ: [55, 45, 45, 55]  # Ciphertext modulus chain
  LogP: [55]              # Key switching modulus
  LogScale: 45            # Scaling factor
  H: 192                  # Hamming weight
  RingType: standard      # Ring type

orion:
  margin: 2               # Security margin
  embedding_method: hybrid # Encoding method
  backend: lattigo        # Backend library
  fuse_modules: true      # Module fusion
  debug: false           # Debug mode
```

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.py`: Basic translation functionality
- `mlp_translation.py`: Neural network example
- Configuration file usage
- Error handling patterns

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test category  
python -m pytest tests/test_core/
python -m pytest tests/test_frontends/test_orion/
```

## Development

### Setting up Development Environment

```bash
git clone <repository-url>
cd orion-heir-translator
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Quality

The project uses:
- **Black**: Code formatting
- **isort**: Import sorting  
- **mypy**: Type checking
- **flake8**: Linting
- **pytest**: Testing

```bash
# Format code
black src/ tests/ examples/

# Type check
mypy src/

# Lint
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure code quality checks pass
5. Submit a pull request

## Design Principles

- **No Mock Objects**: Use real upstream dependencies (Orion, xDSL)
- **Standard Naming**: No "redesigned" or "enhanced" prefixes
- **Clean Separation**: Clear boundaries between components
- **Extensibility**: Easy to add new frontends and operations
- **Standalone**: Self-contained with minimal external dependencies

## License

Apache 2.0 License (see LICENSE file)

## Acknowledgments

- Built on the [xDSL](https://github.com/xdslproject/xdsl) framework
- HEIR dialects inspired by the [HEIR project](https://github.com/google/heir)
- Designed for integration with the Orion FHE library
