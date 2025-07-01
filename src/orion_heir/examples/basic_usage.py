#!/usr/bin/env python3
"""
Strict Orion requirement example.

This example demonstrates how to require the actual Orion library
and fail hard if it's not available.
"""

import torch
from pathlib import Path

from orion_heir import GenericTranslator, FHEOperation
from orion_heir.frontends.orion.scheme_params import (
    OrionSchemeParameters, 
    OrionNotAvailableError,
    create_orion_parameters_strict
)


def create_sample_operations():
    """Create sample FHE operations."""
    return [
        FHEOperation(
            op_type="mul_plain",
            method_name="mul_plain",
            args=[torch.tensor([[1.0, 2.0, 3.0, 4.0]])],
            kwargs={},
            result_var="weight_mul",
            level=3
        ),
        FHEOperation(
            op_type="add_plain",
            method_name="add_plain",
            args=[torch.tensor([[0.5, 0.5, 0.5, 0.5]])],
            kwargs={},
            result_var="bias_add",
            level=3
        )
    ]


def test_orion_required_mode():
    """Test the strict Orion requirement mode."""
    print("🚨 Testing Strict Orion Requirement Mode")
    print("=" * 50)
    
    try:
        # Create scheme parameters that require Orion
        print("📋 Creating scheme parameters with require_orion=True...")
        scheme_params = create_orion_parameters_strict()
        
        print("📊 Attempting to access modulus chain (this will check for Orion)...")
        moduli = scheme_params.ciphertext_modulus_chain
        
        # If we get here, Orion is available
        print(f"✅ Orion is available! Retrieved moduli: {moduli}")
        
        # Continue with translation
        operations = create_sample_operations()
        translator = GenericTranslator()
        
        print("🔄 Translating with actual Orion primes...")
        module = translator.translate(operations, scheme_params, "orion_strict_example")
        
        print("✅ Translation completed successfully with Orion!")
        
    except OrionNotAvailableError as e:
        print(f"❌ HARD FAILURE: {e}")
        print("\n💡 To fix this:")
        print("   1. Install Orion: pip install orion-fhe")
        print("   2. Or set require_orion=False to use fallback primes")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True


def test_orion_optional_mode():
    """Test the optional Orion mode with fallback."""
    print("\n🔄 Testing Optional Orion Mode (with fallback)")
    print("=" * 50)
    
    try:
        # Create scheme parameters that don't require Orion
        print("📋 Creating scheme parameters with require_orion=False...")
        scheme_params = OrionSchemeParameters(
            logN=[12],
            logQ=[55, 45, 45, 55],
            logP=[55],
            logScale=45,
            slots=2048,
            ring_degree=4096,
            backend='lattigo',
            require_orion=False  # Allow fallback
        )
        
        print("📊 Accessing modulus chain (will use fallback if Orion unavailable)...")
        moduli = scheme_params.ciphertext_modulus_chain
        
        print(f"✅ Got moduli: {moduli}")
        
        # Continue with translation
        operations = create_sample_operations()
        translator = GenericTranslator()
        
        print("🔄 Translating with computed primes...")
        module = translator.translate(operations, scheme_params, "orion_optional_example")
        
        print("✅ Translation completed successfully with fallback!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True


def test_manual_orion_check():
    """Manually check for Orion availability."""
    print("\n🔍 Manual Orion Availability Check")
    print("=" * 50)
    
    try:
        import orion.core.orion
        print("✅ Orion library is installed and importable")
        
        # Try to create a simple scheme
        from orion.core.orion import Scheme
        scheme = Scheme()
        print("✅ Orion Scheme can be instantiated")
        
        return True
        
    except ImportError as e:
        print("❌ Orion library is not installed")
        print(f"   Import error: {e}")
        print("\n💡 Install with: pip install orion-fhe")
        
    except Exception as e:
        print(f"❌ Error testing Orion: {e}")
        
    return False


def demonstrate_different_modes():
    """Demonstrate different ways to handle Orion requirement."""
    print("\n📋 Different Orion Requirement Modes")
    print("=" * 50)
    
    # Mode 1: Strict requirement (hard fail)
    print("1️⃣ Strict Mode (require_orion=True):")
    try:
        strict_params = OrionSchemeParameters(
            logN=[12], logQ=[55, 45], logP=[55], logScale=45,
            slots=2048, ring_degree=4096, require_orion=True
        )
        _ = strict_params.ciphertext_modulus_chain
        print("   ✅ Works with Orion")
    except OrionNotAvailableError:
        print("   ❌ Hard fails without Orion")
    
    # Mode 2: Optional with fallback
    print("\n2️⃣ Optional Mode (require_orion=False):")
    try:
        optional_params = OrionSchemeParameters(
            logN=[12], logQ=[55, 45], logP=[55], logScale=45,
            slots=2048, ring_degree=4096, require_orion=False
        )
        _ = optional_params.ciphertext_modulus_chain
        print("   ✅ Works with or without Orion (uses fallback)")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Mode 3: Default behavior (currently optional)
    print("\n3️⃣ Default Mode (no require_orion specified):")
    try:
        default_params = OrionSchemeParameters(
            logN=[12], logQ=[55, 45], logP=[55], logScale=45,
            slots=2048, ring_degree=4096
        )
        _ = default_params.ciphertext_modulus_chain
        print("   ✅ Uses fallback by default")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


def main():
    """Run all Orion requirement tests."""
    print("🚨 Orion Requirement Testing")
    print("=" * 60)
    
    # Check if Orion is available
    orion_available = test_manual_orion_check()
    
    # Test strict mode
    strict_success = test_orion_required_mode()
    
    # Test optional mode
    optional_success = test_orion_optional_mode()
    
    # Demonstrate different modes
    demonstrate_different_modes()
    
    print("\n📊 Summary:")
    print(f"   Orion Available: {'✅' if orion_available else '❌'}")
    print(f"   Strict Mode: {'✅' if strict_success else '❌'}")
    print(f"   Optional Mode: {'✅' if optional_success else '❌'}")
    
    if not orion_available:
        print("\n💡 To install Orion:")
        print("   pip install orion-fhe")
        print("\n🔧 To force hard failure when Orion is missing:")
        print("   scheme_params = OrionSchemeParameters(..., require_orion=True)")
        print("   # or")
        print("   scheme_params = create_orion_parameters_strict()")


if __name__ == "__main__":
    main()
