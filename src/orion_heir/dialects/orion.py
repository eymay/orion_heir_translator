"""
An Orion entry dialect for HEIR.

Just handles the ops not in ORION.
"""

from xdsl.dialects.builtin import FloatAttr, ArrayAttr, f64, TensorType
from xdsl.ir import Dialect
from xdsl.irdl import (
    IRDLOperation,
    ParsePropInAttrDict,
    base,
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
    Linear transformation operation in CKKS (for compatibility with KeyMemRT).

    This operation performs a linear transformation using precomputed diagonal plaintexts.
    It takes a ciphertext input and cleartext weights (diagonals) as inputs.

    Example:
    %result = ckks.linear_transform %ciphertext, %weights : (!ct, tensor<NxMxf64>) -> !ct
    """

    name = "ckks.linear_transform"

    input = operand_def(NewLWECiphertextType)

    # Use cleartext types because BSGS implementation will rotate some of the
    # plaintexts and it's better to decode them live rather than decode,
    # rotate, then re-encode, Best would be to pre-rotate and store, but this
    # depends on the BSGS implementation which Orion doesn't know about.
    weights = operand_def(base(TensorType[f64]))
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
