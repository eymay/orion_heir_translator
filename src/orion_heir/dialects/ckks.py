"""
CKKS dialect implementation for xDSL
Provides CKKS homomorphic encryption operations and attributes
"""

from collections.abc import Sequence
from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr, IndexType, ArrayAttr, DenseArrayBase
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, SSAValue
from xdsl.irdl import (
    BaseAttr,
    IRDLOperation,
    ParameterDef,
    ParsePropInAttrDict,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    traits_def,
    VarConstraint,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import Pure

# Import our custom dialects - this ensures they are loaded when CKKS is loaded
from .lwe import LWE, NewLWECiphertextType, NewLWEPlaintextType
from .polynomial import Polynomial, RingAttr
from .mod_arith import ModArith
from .rns import RNS


@irdl_attr_definition
class SchemeParamAttr(ParametrizedAttribute):
    """
    CKKS scheme parameters attribute.

    Syntax: #ckks.scheme_param<logN = val, Q = [q1, q2, ...], P = [p1, p2, ...], logDefaultScale = val>
    """

    name = "ckks.scheme_param"

    logN: ParameterDef[IntegerAttr]
    Q: ParameterDef[ArrayAttr]  # Array of integers
    P: ParameterDef[ArrayAttr]  # Array of integers
    logDefaultScale: ParameterDef[IntegerAttr]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse CKKS scheme parameters."""
        parser.parse_punctuation("<")
        
        # Parse "logN = value"
        parser.parse_keyword("logN")
        parser.parse_punctuation("=")
        logN_value = parser.parse_integer()
        logN_attr = IntegerAttr.from_int_and_width(logN_value, 64)
        
        parser.parse_punctuation(",")
        
        # Parse "Q = [val1, val2, ...]"
        parser.parse_keyword("Q")
        parser.parse_punctuation("=")
        parser.parse_punctuation("[")
        
        Q_values = []
        # Parse first value
        Q_values.append(parser.parse_integer())
        
        # Parse remaining values
        while parser.parse_optional_punctuation(","):
            Q_values.append(parser.parse_integer())
        
        parser.parse_punctuation("]")
        Q_attrs = [IntegerAttr.from_int_and_width(val, 64) for val in Q_values]
        Q_array = ArrayAttr(Q_attrs)
        
        parser.parse_punctuation(",")
        
        # Parse "P = [val1, val2, ...]"
        parser.parse_keyword("P")
        parser.parse_punctuation("=")
        parser.parse_punctuation("[")
        
        P_values = []
        # Parse first value
        P_values.append(parser.parse_integer())
        
        # Parse remaining values  
        while parser.parse_optional_punctuation(","):
            P_values.append(parser.parse_integer())
        
        parser.parse_punctuation("]")
        P_attrs = [IntegerAttr.from_int_and_width(val, 64) for val in P_values]
        P_array = ArrayAttr(P_attrs)
        
        parser.parse_punctuation(",")
        
        # Parse "logDefaultScale = value"
        parser.parse_keyword("logDefaultScale")
        parser.parse_punctuation("=")
        logDefaultScale_value = parser.parse_integer()
        logDefaultScale_attr = IntegerAttr.from_int_and_width(logDefaultScale_value, 64)
        
        parser.parse_punctuation(">")
        
        return [logN_attr, Q_array, P_array, logDefaultScale_attr]

    def print_parameters(self, printer: Printer) -> None:
        """Print CKKS scheme parameters."""
        printer.print_string("<logN = ")
        printer.print_string(str(self.logN.value.data))
        printer.print_string(", Q = [")
        
        for i, q_val in enumerate(self.Q.data):
            if i > 0:
                printer.print_string(", ")
            printer.print_string(str(q_val.value.data))
        
        printer.print_string("], P = [")
        
        for i, p_val in enumerate(self.P.data):
            if i > 0:
                printer.print_string(", ")
            printer.print_string(str(p_val.value.data))
        
        printer.print_string("], logDefaultScale = ")
        printer.print_string(str(self.logDefaultScale.value.data))
        printer.print_string(">")


@irdl_op_definition
class AddOp(IRDLOperation):
    """
    Addition operation between CKKS ciphertexts.

    Example: %add = ckks.add %arg0, %arg1 : (!ct, !ct) -> !ct
    """

    name = "ckks.add"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWECiphertextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class SubOp(IRDLOperation):
    """
    Subtraction operation between CKKS ciphertexts.

    Example: %sub = ckks.sub %arg0, %arg1 : (!ct, !ct) -> !ct
    """

    name = "ckks.sub"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWECiphertextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class MulOp(IRDLOperation):
    """
    Multiplication operation between CKKS ciphertexts.

    Example: %mul = ckks.mul %arg0, %arg1 : (!ct, !ct) -> !ct1
    """

    name = "ckks.mul"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWECiphertextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class NegateOp(IRDLOperation):
    """
    Negation operation for CKKS ciphertexts.

    Example: %neg = ckks.negate %arg0 : !ct
    """

    name = "ckks.negate"
    T: ClassVar = VarConstraint("T", BaseAttr(NewLWECiphertextType))

    input = operand_def(T)
    result = result_def(T)

    traits = traits_def(Pure())

    assembly_format = "$input attr-dict `:` type($input)"


@irdl_op_definition
class AddPlainOp(IRDLOperation):
    """
    Addition operation between ciphertext and plaintext.

    Example: %add = ckks.add_plain %ct, %pt : (!ct, !pt) -> !ct
    """

    name = "ckks.add_plain"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWEPlaintextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class SubPlainOp(IRDLOperation):
    """
    Subtraction operation between ciphertext and plaintext.

    Example: %sub = ckks.sub_plain %ct, %pt : (!ct, !pt) -> !ct
    """

    name = "ckks.sub_plain"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWEPlaintextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class MulPlainOp(IRDLOperation):
    """
    Multiplication operation between ciphertext and plaintext.

    Example: %mul = ckks.mul_plain %ct, %pt : (!ct, !pt) -> !ct
    """

    name = "ckks.mul_plain"

    lhs = operand_def(NewLWECiphertextType)
    rhs = operand_def(NewLWEPlaintextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)"


@irdl_op_definition
class RelinearizeOp(IRDLOperation):
    """
    Relinearization operation to reduce ciphertext size.

    Example: %relin = ckks.relinearize %ct {from_basis = array<i32: 0, 1, 2>, to_basis = array<i32: 0, 1>} : !ct1 -> !ct
    """

    name = "ckks.relinearize"

    input = operand_def(NewLWECiphertextType)
    result = result_def(NewLWECiphertextType)

    from_basis = prop_def(DenseArrayBase)
    to_basis = prop_def(DenseArrayBase)

    traits = traits_def(Pure())
    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"


@irdl_op_definition
class RescaleOp(IRDLOperation):
    """
    Rescaling operation (CKKS version of modulus switching).

    Example: %rescale = ckks.rescale %ct {to_ring = #ring} : !ct -> !ct2
    """

    name = "ckks.rescale"

    input = operand_def(NewLWECiphertextType)
    result = result_def(NewLWECiphertextType)

    to_ring = prop_def(RingAttr)  # Now properly typed as RingAttr

    traits = traits_def(Pure())
    irdl_options = [ParsePropInAttrDict()]

    assembly_format = "$input attr-dict `:` type($input) `->` type($result)"


@irdl_op_definition
class RotateOp(IRDLOperation):
    """
    Rotation operation for CKKS ciphertexts.

    Example: %rot = ckks.rotate %ct { offset = 1 } : !ct_tensor
    """

    name = "ckks.rotate"

    # Define the type variable as a class variable
    T: ClassVar = VarConstraint("T", BaseAttr(NewLWECiphertextType))

    input = operand_def(T)
    result = result_def(T)  # Same type as input - this allows inference

    offset = prop_def(IntegerAttr)

    traits = traits_def(Pure())
    irdl_options = [ParsePropInAttrDict()]

    # Now the result type can be inferred from the input type
    assembly_format = "$input attr-dict `:` type($input)"


@irdl_op_definition
class ExtractOp(IRDLOperation):
    """
    Extract operation to get a scalar from a tensor ciphertext.

    Example: %ext = ckks.extract %ct, %idx : (!ct_tensor, index) -> !ct_scalar
    """

    name = "ckks.extract"

    input = operand_def(NewLWECiphertextType)
    offset = operand_def(IndexType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$input `,` $offset attr-dict `:` `(` type($input) `,` type($offset) `)` `->` type($result)"


@irdl_op_definition
class MatMulOp(IRDLOperation):
    """
    Matrix multiplication operation for CKKS.
    
    High-level operation that represents linear transformation / matrix multiplication.
    This will be lowered to primitive CKKS operations by a separate pass.

    Example: %result = ckks.matmul %input, %weight : (!ct, !pt) -> !ct
    """

    name = "ckks.matmul"

    input = operand_def(NewLWECiphertextType)
    weight = operand_def(NewLWEPlaintextType)
    result = result_def(NewLWECiphertextType)

    traits = traits_def(Pure())

    assembly_format = "$input `,` $weight attr-dict `:` `(` type($input) `,` type($weight) `)` `->` type($result)"


CKKS = Dialect(
    "ckks",
    [
        AddOp,
        SubOp,
        MulOp,
        NegateOp,
        AddPlainOp,
        SubPlainOp,
        MulPlainOp,
        RelinearizeOp,
        RescaleOp,
        RotateOp,
        ExtractOp,
        MatMulOp
    ],
    [
        SchemeParamAttr,
    ],
)
