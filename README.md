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
```

## Installation

### From Source
```bash
git clone https://github.com/eymay/orion_heir_translator.git
cd orion_heir_translator
pip install -e .
```
Dependencies should be installed automatically.

### Extra dependencies
Orion can be picky on Python version, please install it as explained [here](https://github.com/baahl-nyu/orion?tab=readme-ov-file#install-orion).

## Quick Start
```bash
python examples/mlp.py
python examples/resnet.py
```

