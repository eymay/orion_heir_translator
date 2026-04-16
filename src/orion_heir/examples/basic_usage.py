#!/usr/bin/env python3
"""
Basic usage example for the Orion-HEIR translator.

Demonstrates creating scheme parameters, defining FHE operations,
and translating them to HEIR MLIR.
"""

import torch

from orion_heir import GenericTranslator, FHEOperation
from orion_heir.frontends.orion.scheme_params import OrionSchemeParameters


def create_sample_operations():
    """Create sample FHE operations."""
    return [
        FHEOperation(
            op_type="mul_plain",
            method_name="mul_plain",
            args=[torch.tensor([[1.0, 2.0, 3.0, 4.0]])],
            kwargs={},
            result_var="weight_mul",
            level=3,
        ),
        FHEOperation(
            op_type="add_plain",
            method_name="add_plain",
            args=[torch.tensor([[0.5, 0.5, 0.5, 0.5]])],
            kwargs={},
            result_var="bias_add",
            level=3,
        ),
    ]


def main():
    """Run basic translation example."""
    print("🔧 Basic Orion-HEIR Translation Example")
    print("=" * 50)

    # Create scheme parameters (requires Orion library)
    scheme_params = OrionSchemeParameters(
        logN=[12],
        logQ=[55, 45, 45, 55],
        logP=[55],
        logScale=45,
        slots=2048,
        ring_degree=4096,
        backend="lattigo",
    )

    print(f"✅ Scheme parameters: {scheme_params}")
    print(f"   Modulus chain: {scheme_params.ciphertext_modulus_chain}")

    # Create and translate operations
    operations = create_sample_operations()
    translator = GenericTranslator()

    print("🔄 Translating to HEIR MLIR...")
    module = translator.translate(operations, scheme_params, "basic_example")
    print("✅ Translation completed successfully!")


if __name__ == "__main__":
    main()
