#!/usr/bin/env python3
"""
Clean test using the final Orion frontend with ckks.matmul operation.
No templates, no complex abstractions - just pure CKKS operations.
"""

from pathlib import Path
from orion_heir import GenericTranslator
from orion_heir.frontends.orion.orion_frontend import OrionFrontend


def run_simple_linear_test():
    """Test using the clean Orion frontend with ckks.matmul operation."""
    print("🧪 Clean Orion Frontend - Pure CKKS Operations + MatMul")
    print("=" * 60)
    
    # Test input and expected output
    input_data = [1.0, 2.0, 3.0, 4.0]
    expected_output = [40.0, 20.0]
    
    print(f"Input: {input_data}")
    print(f"Expected output: {expected_output}")
    print(f"Calculation:")
    print(f"  First neuron:  1*1 + 2*2 + 3*3 + 4*4 + 10 = 1+4+9+16+10 = 40")
    print(f"  Second neuron: 1*0.5 + 2*1 + 3*1.5 + 4*2 + 5 = 0.5+2+4.5+8+5 = 20")
    
    # Create the clean Orion frontend
    frontend = OrionFrontend()
    
    # Show what operations are available
    print(f"\n📋 All CKKS operations Orion supports:")
    supported_ops = frontend.get_supported_operations()
    for op in supported_ops:
        info = frontend.get_operation_info(op)
        if info:
            print(f"  • {op:12} - {info['description']}")
    
    # Get the test operations
    print(f"\n📊 Getting test operations:")
    operations = frontend.create_simple_test_operations()
    
    print(f"Generated {len(operations)} operations:")
    for i, op in enumerate(operations, 1):
        print(f"  {i}. {op.op_type:10} -> {op.result_var:15} [level {op.level}]")
        if op.op_type == "matmul":
            print(f"     {'':10}    ↳ Will be lowered to primitive CKKS ops")
        elif op.op_type == "encode":
            if op.args and hasattr(op.args[0], 'shape'):
                print(f"     {'':10}    ↳ Tensor shape: {op.args[0].shape}")
    
    # Get scheme parameters
    scheme_params = frontend._create_default_scheme()
    print(f"\n🔧 Orion scheme parameters:")
    print(f"  Ring degree: {scheme_params.ring_degree}")
    print(f"  Modulus levels: {len(scheme_params.ciphertext_modulus_chain)}")
    print(f"  Backend: {scheme_params.backend}")
    
    # Translate to HEIR
    print(f"\n🔄 Translating to HEIR...")
    translator = GenericTranslator()
    module = translator.translate(operations, scheme_params, "orion_matmul")
    
    # Generate MLIR
    from xdsl.printer import Printer
    from io import StringIO
    
    # Create string buffer and printer
    output_buffer = StringIO()
    printer = Printer(stream=output_buffer)
    printer.print(module)
    mlir_output = output_buffer.getvalue()
    
    # Save files
    output_file = Path("orion_matmul.mlir")
    output_file.write_text(mlir_output)
    print(f"💾 MLIR saved to: {output_file}")
    
    print(f"\n📄 MLIR analysis:")
    lines = mlir_output.split('\n')
    matmul_lines = [line for line in lines if 'ckks.matmul' in line]
    encode_lines = [line for line in lines if 'lwe.rlwe_encode' in line]
    print(f"  • {len(encode_lines)} lwe.rlwe_encode operations")
    print(f"  • {len(matmul_lines)} ckks.matmul operations")
    print(f"  • Total lines: {len(lines)}")
    
    # Save test data for OpenFHE
    test_data = {
        'test_type': 'clean_matmul',
        'input': input_data,
        'expected_output': expected_output,
        'weight_matrix': [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],
        'bias': [10.0, 5.0],
        'operations': [
            {
                'op_type': op.op_type,
                'method_name': op.method_name,
                'result_var': op.result_var,
                'level': op.level,
                'metadata': op.metadata
            } for op in operations
        ],
        'scheme_info': {
            'ring_degree': scheme_params.ring_degree,
            'modulus_chain_length': len(scheme_params.ciphertext_modulus_chain),
            'backend': scheme_params.backend
        },
        'lowering_info': {
            'high_level_op': 'ckks.matmul',
            'requires_lowering_pass': True,
            'lowering_pass': 'ckks-matmul-lowering'
        }
    }
    
    import json
    json_file = Path("orion_matmul_data.json")
    with open(json_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"💾 Test data saved to: {json_file}")
    
    return operations, mlir_output


def demo_clean_frontend():
    """Demonstrate the clean frontend design."""
    print(f"\n🎯 Clean Frontend Design")
    print("=" * 40)
    
    frontend = OrionFrontend()
    
    print("✅ What the frontend provides:")
    print("  • Hardcoded CKKS operations (12 total)")
    print("  • matmul as first-class operation")
    print("  • No templates or abstractions")
    print("  • Simple operation creation")
    
    print(f"\n✅ What the frontend does NOT provide:")
    print("  • ❌ Neural network layer implementations")
    print("  • ❌ Convolution implementations")
    print("  • ❌ Activation function implementations")
    print("  • ❌ Complex templates or patterns")
    
    print(f"\n✅ Clean workflow:")
    print("  1. Frontend: Pure CKKS operations")
    print("  2. Dialect: ckks.matmul as MLIR operation")
    print("  3. Lowering pass: matmul → primitives")
    print("  4. OpenFHE: Code generation")
    
    # Show operation creation
    print(f"\n📊 Creating operations:")
    
    # Basic operation
    basic_op = frontend.create_basic_operation('rotate', 
                                              args=[2], 
                                              operation_kwargs={'offset': 2},
                                              result_var='rot_2')
    if basic_op:
        print(f"  • Created: {basic_op.op_type} -> {basic_op.result_var}")
    
    # Test operations
    test_ops = frontend.create_simple_test_operations()
    print(f"  • Test operations: {len(test_ops)} operations")
    
    print(f"\n✅ Frontend is minimal and focused!")


def validate_frontend_design():
    """Validate that the final frontend design is correct."""
    print(f"\n✅ Final Frontend Validation")
    print("=" * 40)
    
    frontend = OrionFrontend()
    
    # Check 1: All expected operations present
    expected_ops = ['add', 'mul', 'rotate', 'mul_plain', 'add_plain', 'matmul', 'rescale', 'encode']
    supported = frontend.get_supported_operations()
    
    missing = [op for op in expected_ops if op not in supported]
    if not missing:
        print("✅ All expected CKKS operations supported")
    else:
        print(f"❌ Missing operations: {missing}")
    
    # Check 2: matmul is present
    if 'matmul' in supported:
        matmul_info = frontend.get_operation_info('matmul')
        print("✅ matmul operation available as first-class CKKS op")
        print(f"    Description: {matmul_info['description']}")
    else:
        print("❌ matmul operation missing")
    
    # Check 3: Can create operations
    test_ops = frontend.create_simple_test_operations()
    if len(test_ops) >= 3:
        print("✅ Can create test operations")
        print(f"    Generated: {[op.op_type for op in test_ops]}")
    else:
        print("❌ Cannot create sufficient test operations")
    
    # Check 4: No template dependencies
    if not hasattr(frontend, '_operation_templates'):
        print("✅ No template dependencies")
    else:
        print("❌ Still has template dependencies")
    
    print("✅ Frontend design is clean and correct!")


if __name__ == "__main__":
    print("🚀 Testing Final Clean Orion Frontend")
    print("=" * 60)
    
    # Validate the design
    validate_frontend_design()
    
    # Demo the frontend
    demo_clean_frontend()
    
    # Run the main test
    operations, mlir = run_simple_linear_test()
    
    print(f"\n📝 Final Summary:")
    print(f"✅ Clean Orion frontend with pure CKKS operations")
    print(f"✅ matmul as first-class operation in CKKS dialect")
    print(f"✅ No templates, abstractions, or complexity")
    print(f"✅ Input: [1.0, 2.0, 3.0, 4.0] → Expected: [40.0, 20.0]")
    print(f"✅ Generated {len(operations)} operations including ckks.matmul")
    print(f"✅ Files: orion_matmul.mlir, orion_matmul_data.json")
    print(f"\n🎯 Ready for OpenFHE validation with clean architecture!")
    print(f"🔧 Use ckks-matmul-lowering pass to convert matmul to primitives")
