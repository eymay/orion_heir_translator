# Orion-HEIR Translator

Python based translator that connects Orion Frontend to HEIR backends. It reads internal compilation result of [Orion](https://github.com/baahl-nyu/orion) and constructs an [xDSL](https://github.com/xdslproject/xdsl) based MLIR representation that is compatible with [HEIR](https://github.com/google/heir).

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

### From Source
```bash
git clone https://github.com/eymay/orion_heir_translator.git
cd orion_heir_translator
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

