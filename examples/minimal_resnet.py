#!/usr/bin/env python3
"""
Lightweight Orion ResNet Demo - Using Simple Polynomial Activations

This demonstrates creating a ResNet with lightweight polynomial activations
instead of heavy ReLU decomposition, making it much faster to compile and translate.
"""

import torch
import torch.nn as nn
from pathlib import Path
from io import StringIO

# Import our translator components
from orion_heir import GenericTranslator
from orion_heir.frontends.orion.orion_frontend import OrionFrontend
from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters, OrionNotAvailableError
from xdsl.printer import Printer

# Try to import Orion
try:
    import orion
    import orion.models as models
    import orion.nn as on  # This is the correct import for Orion layers
    from orion.core.utils import get_cifar_datasets
    ORION_AVAILABLE = True
    print("✅ Orion imported successfully")
except ImportError as e:
    print(f"⚠️ Orion not available: {e}")
    ORION_AVAILABLE = False


def create_lightweight_resnet():
    """Create a lightweight ResNet using simple polynomial activations."""
    if not ORION_AVAILABLE:
        print("❌ Orion not available")
        return None
    
    try:
        print("🔧 Creating lightweight ResNet with polynomial activations...")
        
        class LightweightResNet(on.Module):
            """Ultra-lightweight ResNet with polynomial activations and various operations."""
            def __init__(self):
                super().__init__()
                # Convolutional backbone
                self.conv1 = on.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
                self.bn1 = on.BatchNorm2d(8)
                self.act1 = on.Quad()  # x^2 activation
                
                # Second conv layer  
                self.conv2 = on.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
                self.bn2 = on.BatchNorm2d(16)
                
                # Mix of polynomial activations
                self.act2 = on.SiLU(degree=7)  # Low-degree SiLU polynomial
                
                # Additional polynomial layers for demonstration
                self.conv3 = on.Conv2d(16, 16, kernel_size=3, padding=1, bias=False)
                self.bn3 = on.BatchNorm2d(16)
                self.act3 = on.Sigmoid(degree=15)  # Sigmoid polynomial
                
                # Add some manual polynomial activations
                self.poly1 = on.Activation([0.5, 0.0, 0.3, 0.0, 0.1])  # Custom polynomial: 0.5 + 0.3x² + 0.1x⁴
                
                # Chebyshev polynomial for specific function approximation
                self.cheby1 = on.Chebyshev(degree=9, fn=lambda x: torch.tanh(x))  # Tanh approximation
                
                # Global operations
                self.avgpool = on.AdaptiveAvgPool2d((1, 1))
                self.flatten = on.Flatten()
                
                # Final layers with more polynomials
                self.linear1 = on.Linear(16, 8)
                self.act_final = on.GELU()  # GELU polynomial (degree 31 by default)
                self.linear2 = on.Linear(8, 10)
            
            def forward(self, x):
                # Conv + BN + Quad
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.act1(x)  # x^2
                
                # Conv + BN + SiLU polynomial
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.act2(x)  # SiLU polynomial (degree 7)
                
                # Conv + BN + Sigmoid polynomial
                x = self.conv3(x)
                x = self.bn3(x)
                x = self.act3(x)  # Sigmoid polynomial (degree 15)
                
                # Custom polynomial activation
                x = self.poly1(x)  # Custom coefficients
                
                # Chebyshev polynomial
                x = self.cheby1(x)  # Tanh approximation
                
                # Global pooling and classification
                x = self.avgpool(x)
                x = self.flatten(x)
                x = self.linear1(x)
                x = self.act_final(x)  # GELU polynomial
                x = self.linear2(x)
                
                return x
        
        return LightweightResNet()
        
    except Exception as e:
        print(f"❌ Failed to create lightweight ResNet: {e}")
        
        # Fallback: Try even simpler model
        try:
            print("🔧 Falling back to ultra-minimal model...")
            
            class MinimalCNN(on.Module):
                def __init__(self):
                    super().__init__()
                    # Just one conv + activation + linear
                    self.conv = on.Conv2d(3, 8, kernel_size=3, padding=1)
                    self.act = on.Quad()  # Simple x^2
                    self.avgpool = on.AdaptiveAvgPool2d((1, 1))
                    self.flatten = on.Flatten()
                    self.linear = on.Linear(8, 10)
                
                def forward(self, x):
                    x = self.conv(x)
                    x = self.act(x)
                    x = self.avgpool(x)
                    x = self.flatten(x)
                    x = self.linear(x)
                    return x
            
            return MinimalCNN()
            
        except Exception as e2:
            print(f"❌ Even minimal model failed: {e2}")
            return None


def create_fhe_scheme_parameters():
    """Create simpler FHE scheme parameters for lightweight ResNet."""
    try:
        # Use smaller parameters for faster compilation
        return OrionSchemeParameters(
            logN=14,  # Smaller ring degree for speed
            logQ=[45, 35, 35, 35, 35],  # Shorter modulus chain
            logP=[45],
            logScale=35,
            slots=8192,  # Fewer slots
            ring_degree=16384,
            backend='lattigo',
            require_orion=False
        )
    except OrionNotAvailableError:
        # Even simpler fallback
        return OrionSchemeParameters(
            logN=13, 
            logQ=[40, 30, 30, 40], 
            logP=[40], 
            logScale=30,
            slots=4096, 
            ring_degree=8192, 
            backend='lattigo', 
            require_orion=False
        )


def convert_module_to_string(module):
    """Convert XDSL ModuleOp to string using the printer."""
    output_buffer = StringIO()
    printer = Printer(stream=output_buffer)
    printer.print(module)
    return output_buffer.getvalue()


def extract_orion_operations(model, input_tensor):
    """Extract actual operations from Orion's fit and compile process."""
    operations = []
    
    if not ORION_AVAILABLE:
        print("❌ Orion not available - cannot extract real operations")
        return operations
    
    try:
        print("🔧 Initializing Orion with lightweight scheme...")
        # Use simpler scheme for faster compilation
        orion.init_scheme({
            'ckks_params': {
                'LogN': 14,  # Smaller for speed
                'LogQ': [45, 35, 35, 35, 35],  # Shorter chain
                'LogP': [45],
                'LogScale': 35,
                'H': 64,  # Smaller H for speed
                'RingType': 'standard'
            },
            'orion': {
                'margin': 1.5,  # Smaller margin for speed
                'embedding_method': 'hybrid',
                'backend': 'lattigo',
                'fuse_modules': True,  # Enable fusion for efficiency
                'debug': False  # Disable debug for speed
            }
        })
        
        print("🔧 Running orion.fit() (should be much faster)...")
        orion.fit(model, input_tensor)
        
        print("🔧 Running orion.compile() (lightweight model)...")
        input_level = orion.compile(model)
        
        print("🔍 Extracting operations from compiled model...")
        # Use OrionFrontend to extract the actual operations
        frontend = OrionFrontend()
        operations = frontend.extract_operations(model)
        
        print(f"✅ Extracted {len(operations)} operations (much fewer than complex ResNet)")
        
        # Show operation summary
        if operations:
            print("📋 Lightweight Operations:")
            op_counts = {}
            for op in operations:
                op_type = op.op_type
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
            for op_type, count in sorted(op_counts.items()):
                print(f"   {op_type:15}: {count}")
        
        return operations
        
    except Exception as e:
        print(f"❌ Error extracting operations: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_lightweight_resnet_demo():
    """Run the lightweight ResNet demo."""
    print("🚀 Lightweight Orion ResNet Demo - Polynomial Activations")
    print("=" * 65)
    
    # Step 1: Create lightweight model
    print("\n1️⃣ Creating Lightweight ResNet")
    print("-" * 30)
    
    model = create_lightweight_resnet()
    if model is None:
        print("❌ Could not create any Orion model")
        return False
        
    print(f"✅ Created model: {type(model).__name__}")
    
    # Check if this is an Orion model
    if hasattr(model, 'set_scheme'):
        print("✅ Model is Orion-compatible")
    else:
        print("❌ Model is NOT Orion-compatible")
        return False
    
    # Show model structure
    print(f"\n📋 Extended Model Structure:")
    for name, layer in model.named_modules():
        if name:  # Skip the root module
            layer_type = layer.__class__.__name__
            print(f"   {name:15}: {layer_type}")
            if layer_type == 'Quad':
                print(f"   {'':15}  └─ Simple x^2 activation (1 level)")
            elif layer_type == 'SiLU':
                degree = getattr(layer, 'degree', 'unknown')
                print(f"   {'':15}  └─ SiLU polynomial (degree {degree})")
            elif layer_type == 'Sigmoid':
                degree = getattr(layer, 'degree', 'unknown')
                print(f"   {'':15}  └─ Sigmoid polynomial (degree {degree})")
            elif layer_type == 'Activation':
                coeffs = getattr(layer, 'coeffs', [])
                print(f"   {'':15}  └─ Custom polynomial (degree {len(coeffs)-1 if coeffs else 'unknown'})")
            elif layer_type == 'Chebyshev':
                degree = getattr(layer, 'degree', 'unknown')
                fn_name = getattr(layer, 'fn', lambda x: x).__name__ if hasattr(getattr(layer, 'fn', None), '__name__') else 'tanh'
                print(f"   {'':15}  └─ Chebyshev {fn_name} approx (degree {degree})")
            elif layer_type == 'GELU':
                degree = getattr(layer, 'degree', 31)  # GELU default
                print(f"   {'':15}  └─ GELU polynomial (degree {degree})")
                
    print(f"\n💡 Polynomial Operations Expected:")
    print(f"   • Quad: x^2 (1 multiplication)")
    print(f"   • SiLU: Polynomial approximation of x*sigmoid(x)")
    print(f"   • Sigmoid: Polynomial approximation of 1/(1+exp(-x))")
    print(f"   • Custom: User-defined polynomial coefficients")
    print(f"   • Chebyshev: Optimal polynomial approximation")
    print(f"   • GELU: Polynomial approximation of GELU function")
    
    # Initialize with small weights
    with torch.no_grad():
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.1)
            else:
                nn.init.uniform_(param, -0.1, 0.1)
    
    # Step 2: Create sample input
    print("\n2️⃣ Creating Sample Input")
    print("-" * 30)
    
    input_tensor = torch.randn(1, 3, 32, 32) * 0.3  # Even smaller for speed
    print(f"✅ Input shape: {input_tensor.shape}")
    print(f"📊 Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    # Step 3: Run PyTorch reference
    print("\n3️⃣ PyTorch Reference Forward Pass")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        pytorch_output = model(input_tensor)
    
    print(f"✅ PyTorch output shape: {pytorch_output.shape}")
    print(f"📊 Predicted class: {pytorch_output.argmax().item()}")
    print(f"📊 Output logits (first 5): {pytorch_output.flatten()[:5].tolist()}")
    
    # Step 4: Extract operations (should be much faster)
    print("\n4️⃣ Extracting Lightweight Operations")
    print("-" * 30)
    
    fhe_operations = extract_orion_operations(model, input_tensor)
    
    if not fhe_operations:
        print("❌ No operations extracted")
        return False
    
    # Step 5: Create FHE scheme parameters
    print("\n5️⃣ Setting Up Lightweight FHE Scheme")
    print("-" * 30)
    
    scheme_params = create_fhe_scheme_parameters()
    print(f"✅ Ring degree: {scheme_params.ring_degree}")
    print(f"✅ Slots: {scheme_params.slots}")
    print(f"✅ Modulus chain levels: {len(scheme_params.ciphertext_modulus_chain)}")
    
    # Step 6: Translate to HEIR MLIR
    print("\n6️⃣ Translating to HEIR MLIR")
    print("-" * 30)
    
    try:
        translator = GenericTranslator()
        mlir_module = translator.translate(
            fhe_operations, 
            scheme_params, 
            function_name="lightweight_resnet"
        )
        print("✅ Translation to HEIR completed successfully!")
        
    except Exception as e:
        print(f"❌ Translation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 7: Generate MLIR string output
    print("\n7️⃣ Generating MLIR Output")
    print("-" * 30)
    
    try:
        mlir_output = convert_module_to_string(mlir_module)
        print("✅ Successfully converted ModuleOp to string")
        
    except Exception as e:
        print(f"❌ Error converting to string: {e}")
        return False
    
    # Step 8: Save to file
    print("\n8️⃣ Saving MLIR Output")
    print("-" * 30)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "lightweight_resnet.mlir"
    
    try:
        output_file.write_text(mlir_output)
        print(f"✅ MLIR saved to: {output_file}")
        print(f"📏 Output size: {len(mlir_output)} characters")
        
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False
    
    # Step 9: Analyze the output
    print("\n9️⃣ MLIR Analysis")
    print("-" * 30)
    
    lines = mlir_output.split('\n')
    operation_lines = [line for line in lines if any(op in line for op in ['ckks.', 'lwe.', 'arith.', 'func.'])]
    
    print(f"📊 Lightweight MLIR Statistics:")
    print(f"   Total lines: {len(lines)}")
    print(f"   Operation lines: {len(operation_lines)}")
    
    # Step 10: Summary
    print("\n🔟 Lightweight Demo Summary")
    print("-" * 30)
    
    print("✅ Demonstrated extended polynomial approach:")
    print("   • Multiple polynomial activation types")
    print("   • Quad (x^2), SiLU, Sigmoid, GELU polynomials")
    print("   • Custom polynomial with user coefficients")
    print("   • Chebyshev optimal approximations")
    print("   • Range of polynomial degrees (7, 9, 15, 31)")
    print("   • Real Orion compilation and HEIR translation")
    
    print(f"\n📁 Generated files:")
    print(f"   • {output_file}")
    
    print(f"\n💡 Extended Benefits:")
    print(f"   • Quad: Only 1 multiplication, 1 level consumed")
    print(f"   • SiLU (deg 7): ~3 levels, smooth activation")
    print(f"   • Sigmoid (deg 15): ~4 levels, classic S-curve")
    print(f"   • Custom polynomial: User-controlled coefficients")
    print(f"   • Chebyshev: Optimal approximation for any function")
    print(f"   • GELU: Modern activation with polynomial approx")
    print(f"   • All much faster than complex ReLU decomposition")
    print(f"   • Demonstrates full range of Orion polynomial capabilities")
    
    return True


def main():
    """Main function."""
    if not ORION_AVAILABLE:
        print("⚠️ This demo requires Orion to be installed")
        return 1
        
    success = run_lightweight_resnet_demo()
    
    if success:
        print(f"\n🎉 Lightweight ResNet demo completed successfully!")
        return 0
    else:
        print(f"\n❌ Demo failed. Check the error messages above.")
        return 1


if __name__ == "__main__":
    exit(main())
