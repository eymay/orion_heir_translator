# Orion-HEIR Translator

Python based translator that connects Orion Frontend to HEIR backends, 
originally created by @eymay, with modifications by @j2kun and @alexanderviand.

The translator reads the internal compilation result of [Orion](https://github.com/baahl-nyu/orion) and constructs an [xDSL](https://github.com/xdslproject/xdsl)-based MLIR representation that is compatible with [HEIR](https://github.com/google/heir). 

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

### Prerequisites (required for Orion fit/compile)
Orion requires native dependencies (Go toolchain + system crypto/build libs), 
see also Orion's upstream installation notes in the
[Orion repository](https://github.com/baahl-nyu/orion).

On Ubuntu/Debian:
```bash
sudo apt update && sudo apt install -y \
  build-essential pkg-config libgmp-dev libssl-dev golang-go
```

On macOS (Homebrew):
```bash
brew update
brew install go pkg-config openssl@3 gmp
```

### Installing Orion-HEIR-Translator from Source
```bash
git clone https://github.com/alexanderviand/orion_heir_translator.git
cd orion_heir_translator
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```
Dependencies should be installed automatically.

## Quick Start
```bash
python3 examples/mlp.py
python3 examples/resnet.py
```
