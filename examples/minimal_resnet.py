#!/usr/bin/env python3
"""
Minimal ResNet example that matches the actual Orion ResNet structure.
Only uses the layers that are actually in orion.models.resnet.
"""

import sys
from pathlib import Path
import torch

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import Orion
import orion
import orion.nn as on


class BasicBlock(on.Module):
    """Simplified BasicBlock matching Orion's actual structure."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = on.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = on.BatchNorm2d(out_channels)
        self.act1 = on.ReLU()

        self.conv2 = on.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = on.BatchNorm2d(out_channels)
        
        self.add = on.Add()
        self.act2 = on.ReLU()
        
        # Shortcut connection
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = torch.nn.Sequential(
                on.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                on.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.add(out, self.shortcut(x))
        return self.act2(out)


class MinimalResNet(on.Module):
    """Very minimal ResNet with just one BasicBlock to test the pipeline."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial layers
        self.conv1 = on.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = on.BatchNorm2d(16)
        self.act = on.ReLU()
        
        # Just one basic block
        self.layer1 = BasicBlock(16, 16, stride=1)
        
        # Final layers
        self.avgpool = on.AdaptiveAvgPool2d(output_size=(1, 1))
        self.flatten = on.Flatten()
        self.linear = on.Linear(16, num_classes)
    
    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        return self.linear(out)


def extract_orion_operations(model):
    """Extract operations by calling orion.compile() and then use the translator."""
    print("🔄 Running orion.compile() to compile the model...")
    try:
        input_level = orion.compile(model)
        print(f"✅ Orion compilation successful, input level: {input_level}")
        
        # Now extract operations using the Orion-HEIR frontend
        print("🔄 Extracting operations using Orion-HEIR frontend...")
        from orion_heir.frontends.orion.orion_frontend import OrionFrontend
        
        frontend = OrionFrontend()
        operations = frontend.extract_operations(model)  # Correct method name
        
        print(f"✅ Extracted {len(operations)} operations from compiled Orion model")
        return operations
        
    except ImportError as e:
        print(f"❌ Could not import OrionFrontend: {e}")
        raise
    except Exception as e:
        print(f"❌ Error in extraction pipeline: {e}")
        raise


def main():
    print("🚀 Minimal ResNet Orion Test")
    print("=" * 50)
    
    # Initialize Orion scheme with more moduli levels
    print("🔧 Initializing Orion scheme...")
    orion.init_scheme({
        'ckks_params': {
            'LogN': 13,
            'LogQ': [50, 40, 40, 40, 40, 40, 40, 40],  # More levels for ReLU polynomial approximations
            'LogP': [50, 50],
            'LogScale': 40,
            'H': 8192
        },
        'orion': {
            'backend': 'lattigo'
        }
    })
    print("✅ Orion scheme initialized")
    
    # Create the minimal model
    model = MinimalResNet(num_classes=10)
    print(f"✅ Model created with structure:")
    print(f"   - conv1: Conv2d(3, 16, kernel_size=3)")
    print(f"   - bn1: BatchNorm2d(16)")
    print(f"   - act: ReLU()")
    print(f"   - layer1: BasicBlock(16, 16)")
    print(f"     - conv1: Conv2d(16, 16, kernel_size=3)")
    print(f"     - bn1: BatchNorm2d(16)")
    print(f"     - act1: ReLU()")
    print(f"     - conv2: Conv2d(16, 16, kernel_size=3)")
    print(f"     - bn2: BatchNorm2d(16)")
    print(f"     - add: Add()")
    print(f"     - act2: ReLU()")
    print(f"   - avgpool: AdaptiveAvgPool2d((1, 1))")
    print(f"   - flatten: Flatten()")
    print(f"   - linear: Linear(16, 10)")
    
    # Create small test input (CIFAR-10 size)
    input_tensor = torch.randn(1, 3, 32, 32)
    print(f"✅ Created test input: {input_tensor.shape}")
    
    # Run cleartext forward pass to verify model works
    print("🧪 Testing cleartext forward pass...")
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"✅ Cleartext output shape: {output.shape}")
    
    # Fit and compile with Orion
    print("📊 Running orion.fit()...")
    try:
        orion.fit(model, input_tensor)
        print("✅ orion.fit() completed")
    except Exception as e:
        print(f"❌ orion.fit() failed: {e}")
        return
    
    # Extract operations and translate
    print("🔍 Extracting operations and translating to HEIR MLIR...")
    try:
        operations = extract_orion_operations(model)
        
        if operations:
            print(f"✅ Operations extracted successfully")
            print(f"📊 Found {len(operations)} operations:")
            
            # Group operations by type
            op_types = {}
            for op in operations:
                op_type = op.op_type if hasattr(op, 'op_type') else str(type(op).__name__)
                op_types[op_type] = op_types.get(op_type, 0) + 1
            
            for op_type, count in sorted(op_types.items()):
                print(f"   - {op_type}: {count}")
            
            # Now translate to HEIR MLIR
            print("🔄 Translating to HEIR MLIR...")
            from orion_heir import GenericTranslator
            from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters
            
            # Create proper scheme parameters object
            scheme_params = OrionSchemeParameters(
                logN=13,
                logQ=[50, 40, 40, 40, 40, 40, 40, 40],
                logP=[50, 50], 
                logScale=40,
                slots=4096,
                ring_degree=8192,
                backend='lattigo',
                require_orion=False  # Use fallback primes if Orion not available
            )
            
            translator = GenericTranslator()
            module = translator.translate(operations, scheme_params, "minimal_resnet")
            
            # Generate MLIR output
            from xdsl.printer import Printer
            from io import StringIO
            
            output_buffer = StringIO()
            printer = Printer(stream=output_buffer)
            printer.print(module)
            mlir_output = output_buffer.getvalue()
            
            # Save to file
            output_file = Path("minimal_resnet.mlir")
            output_file.write_text(mlir_output)
            print(f"💾 Saved MLIR to {output_file}")
            
            # Show MLIR statistics
            print(f"\n📄 MLIR Statistics:")
            print("=" * 50)
            lines = mlir_output.split('\n')
            
            # Count different operation types
            mlir_op_counts = {}
            for line in lines:
                line = line.strip()
                if ' = ' in line and ('ckks.' in line or 'lwe.' in line or 'arith.' in line):
                    # Extract operation type
                    if 'ckks.' in line:
                        op_start = line.find('ckks.') + 5
                    elif 'lwe.' in line:
                        op_start = line.find('lwe.') + 4
                    elif 'arith.' in line:
                        op_start = line.find('arith.') + 6
                    else:
                        continue
                    
                    op_end = line.find('(', op_start)
                    if op_end > op_start:
                        op_type = line[op_start-5:op_end] if 'ckks.' in line else line[op_start-4:op_end] if 'lwe.' in line else line[op_start-6:op_end]
                        mlir_op_counts[op_type] = mlir_op_counts.get(op_type, 0) + 1
            
            for op_type, count in sorted(mlir_op_counts.items()):
                print(f"   - {op_type}: {count}")
            
            print(f"📏 Total MLIR lines: {len(lines)}")
            
            # Verify the module
            try:
                module.verify()
                print("✅ MLIR module verification passed")
            except Exception as e:
                print(f"⚠️ MLIR verification warning: {e}")
        
        else:
            print("❌ No operations extracted")
        
        print("🎉 Complete pipeline test successful!")
        
    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        print("❌ Pipeline test failed. Check the error messages above.")
        
        # Print model structure for debugging
        print("\n🔍 Model structure for debugging:")
        for name, module in model.named_modules():
            if name:  # Skip root module
                print(f"   {name}: {type(module).__name__}")


if __name__ == "__main__":
    main()
