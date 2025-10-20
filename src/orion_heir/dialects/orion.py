"""
An Orion entry dialect for HEIR.

Just handles the ops not in ORION.
"""

from xdsl.dialects.builtin import FloatAttr, ArrayAttr, f64
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    irdl_op_definition,
    operand_def,
    prop_def,
    result_def,
    traits_def,
)

from xdsl.traits import Pure

# Ensure LWE is loaded
from .lwe import NewLWECiphertextType, NewLWEPlaintextType


@irdl_op_definition
class LinearTransformOp(IRDLOperation):
    """
    Linear transformation operation in ORION.

    This operation performs a linear transformation using precomputed diagonal plaintexts.
    It takes a ciphertext input and plaintext weights (diagonals) as inputs.

    Example:
    %result = orion.linear_transform %ciphertext, %weights : (!ct, !pt) -> !ct
    """

    name = "orion.linear_transform"

    # Two inputs: input ciphertext and plaintext weights
    input = operand_def(NewLWECiphertextType)
    weights = operand_def(NewLWEPlaintextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$input `,` $weights attr-dict `:` `(` type($input) `,` type($weights) `)` `->` type($result)"


@irdl_op_definition
class ChebyshevOp(IRDLOperation):
    """
    ORION Chebyshev polynomial evaluation operation.

    Evaluates a Chebyshev polynomial series on a ciphertext using pre-computed
    coefficients. This operation directly maps to OpenFHE's EvalChebyshevSeries.

    Syntax:
    ```mlir
    %result = orion.chebyshev %input {
        coefficients = [1.0, 0.0, -0.33333, 0.0, 0.2, ...],
        domain_start = -1.0,
        domain_end = 1.0
    } : (!orion.ciphertext) -> !orion.ciphertext
    ```

    Attributes:
    - coefficients: Array of Chebyshev polynomial coefficients (required)
    - domain_start: Start of approximation domain (default: -1.0)
    - domain_end: End of approximation domain (default: 1.0)

    The coefficients correspond to the Chebyshev series:
    f(x) = c₀T₀(x) + c₁T₁(x) + c₂T₂(x) + ... + cₙTₙ(x)

    where Tᵢ(x) are Chebyshev polynomials of the first kind.
    """

    name = "orion.chebyshev"

    # Input ciphertext to evaluate polynomial on
    input = operand_def(NewLWECiphertextType)

    # Chebyshev polynomial coefficients (required)
    coefficients = prop_def(ArrayAttr[FloatAttr])

    # Domain of approximation (optional, defaults to [-1, 1])
    domain_start = prop_def(FloatAttr, default=FloatAttr(-1.0, f64))
    domain_end = prop_def(FloatAttr, default=FloatAttr(1.0, f64))

    # Result ciphertext
    result = result_def(NewLWECiphertextType)

    irdl_options = [ParsePropInAttrDict()]
    assembly_format = "$input attr-dict `:` `(` type($input)  `)` `->` type($result)"

    @property
    def degree(self) -> int:
        """Get the degree of the Chebyshev polynomial."""
        return len(self.coefficients.data) - 1

    def get_coefficients(self) -> list[float]:
        """Get the coefficients as a Python list."""
        return [float(attr.value) for attr in self.coefficients.data]

    def get_domain(self) -> tuple[float, float]:
        """Get the approximation domain as (start, end) tuple."""
        return (float(self.domain_start.value), float(self.domain_end.value))


ORION = Dialect(
    "orion",
    [
        LinearTransformOp,
        ChebyshevOp,
    ],
    [
       # no attributes
    ],
)
