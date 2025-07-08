#!/usr/bin/env python3
"""
Minimal test to see what Orion actually does - no translation, just print Orion's IR.
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion (assuming it's available)
import orion
import orion.nn as on


class MinimalModel(on.Module):
    """Simple single layer model."""
    
    def __init__(self):
        super().__init__()
        self.fc1 = on.Linear(784, 128)

    def forward(self, x):
        return self.fc1(x)


def main():
    print("🚀 Minimal Orion Test - Show Orion's Actual IR")
    print("=" * 50)
    
    # Initialize Orion scheme
    orion.init_scheme({
        'ckks_params': {
            'LogN': 13,
            'LogQ': [55, 45, 45, 55],
            'LogP': [55],
            'LogScale': 45
        },
        'orion': {
            'backend': 'lattigo'
        }
    })
    print("✅ Orion scheme initialized")
    
    # Create the model
    model = MinimalModel()
    print(f"✅ Model: fc1 = Linear(784, 128)")
    
    # Create input
    x = torch.randn(1, 784)
    print(f"✅ Input shape: {x.shape}")
    
    # Test cleartext forward
    model.eval()
    with torch.no_grad():
        output = model(x)
    print(f"✅ Cleartext output shape: {output.shape}")
    
    # Fit and compile the model
    print(f"\n🔧 Orion fit and compile...")
    orion.fit(model, x)
    input_level = orion.compile(model)
    print(f"✅ Compiled successfully, input level: {input_level}")
    
    # Print what Orion created during compile
    print(f"\n📋 Orion Compilation Results:")
    print(f"=" * 40)
    
    for name, layer in model.named_modules():
        if hasattr(layer, 'diagonals') or hasattr(layer, 'transform_ids'):
            print(f"\nLayer: {name}")
            print(f"  Type: {layer.__class__.__name__}")
            print(f"  Level: {getattr(layer, 'level', 'N/A')}")
            print(f"  Has weight: {hasattr(layer, 'weight')}")
            print(f"  Has bias: {hasattr(layer, 'bias')}")
            print(f"  Weight shape: {layer.weight.shape if hasattr(layer, 'weight') else 'N/A'}")
            print(f"  Bias shape: {layer.bias.shape if hasattr(layer, 'bias') else 'N/A'}")
            
            if hasattr(layer, 'diagonals'):
                print(f"  Diagonals: {len(layer.diagonals) if layer.diagonals else 0} blocks")
                if layer.diagonals:
                    print(f"    Keys: {list(layer.diagonals.keys())[:3]}{'...' if len(layer.diagonals) > 3 else ''}")
            
            if hasattr(layer, 'transform_ids'):
                print(f"  Transform IDs: {len(layer.transform_ids) if layer.transform_ids else 0}")
                if layer.transform_ids:
                    print(f"    Keys: {list(layer.transform_ids.keys())[:3]}{'...' if len(layer.transform_ids) > 3 else ''}")
            
            if hasattr(layer, 'output_rotations'):
                print(f"  Output rotations: {layer.output_rotations}")
            
            if hasattr(layer, 'on_bias_ptxt'):
                print(f"  Bias plaintext: {'Created' if layer.on_bias_ptxt is not None else 'None'}")
    
    # Switch to HE mode and see what operations Orion would do
    print(f"\n🔄 Switching to HE mode...")
    model.he()
    
    # Create encrypted input
    vec_ptxt = orion.encode(x, input_level)
    vec_ctxt = orion.encrypt(vec_ptxt)
    print(f"✅ Created encrypted input")
    
    # THIS IS THE KEY: Run Orion inference and see what it actually does
    print(f"\n🎯 Orion HE Inference Operations:")
    print(f"=" * 40)
    
    # For now, just show that we have everything set up
    print(f"Input ciphertext level: {vec_ctxt.level()}")
    print(f"Input ciphertext slots: {vec_ctxt.slots()}")
    
    # The actual Orion forward pass would use:
    # - The compiled diagonals (layer.diagonals)
    # - The transform IDs (layer.transform_ids) 
    # - The linear transform evaluator
    # - Rotation operations for output accumulation
    # - Bias addition using on_bias_ptxt
    
    print(f"\n🎯 Extracting Orion Operations:")
    print(f"=" * 40)
    
    # Now extract the correct operations from the compiled model
    from orion_heir.frontends.orion.orion_frontend import OrionFrontend
    frontend = OrionFrontend()
    
    operations = frontend.extract_operations(model)
    print(f"✅ Extracted {len(operations)} operations:")
    
    for i, op in enumerate(operations):
        print(f"  {i+1}. {op.op_type:15} -> {op.result_var}")
        if op.metadata:
            desc = op.metadata.get('operation', '')
            if desc:
                print(f"     {'':15}    ↳ {desc}")
    
    # Now translate to MLIR
    print(f"\n🔄 Translating to HEIR MLIR...")
    from orion_heir import GenericTranslator
    
    scheme_params = frontend._create_default_scheme()
    translator = GenericTranslator()
    module = translator.translate(operations, scheme_params, "minimal_linear")
    
    # Generate MLIR output
    from xdsl.printer import Printer
    from io import StringIO
    
    output_buffer = StringIO()
    printer = Printer(stream=output_buffer)
    printer.print(module)
    mlir_output = output_buffer.getvalue()
    
    Path("minimal_linear.mlir").write_text(mlir_output)
    print("💾 Saved to minimal_linear.mlir")
    
    # Show the MLIR
    print(f"\n📄 Generated MLIR:")
    print("=" * 50)
    lines = mlir_output.split('\n')
    for line in lines:
        if line.strip():
            print(line)
    
    print(f"\n✅ Translation complete!")
    print(f"   Operations: {[op.op_type for op in operations]}")
    print(f"   Based on: {len(model.fc1.diagonals)} diagonals, {model.fc1.output_rotations} rotations")
    
    return 0


if __name__ == "__main__":
    exit(main())
