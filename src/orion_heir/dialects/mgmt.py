"""
MGMT (Management) dialect implementation for xDSL
Provides management operations for plaintext initialization with HE attributes
"""

from collections.abc import Sequence

from typing import ClassVar

from xdsl.dialects.builtin import IntegerAttr, StringAttr, AnyTensorType
from xdsl.ir import Attribute, Dialect, SSAValue
from xdsl.irdl import (
    BaseAttr,
    VarConstraint,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    traits_def,
    opt_attr_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    Pure,
    SameOperandsAndResultType,
)


@irdl_op_definition
class InitOp(IRDLOperation):
    """
    Init the plaintext with mgmt attributes.

    This is a scheme-agnostic operation that initializes the plaintext with `mgmt` attributes.
    Plaintext has multiple sources, e.g. function argument, arith.constant, tensor.empty, etc.
    However, they may have multiple uses in HE circuit and the level/scale information for them
    may be different, so we could not annotate them with `mgmt` attributes directly, as they
    could not have more than one annotation.

    Also, `mgmt` attributes annotated on them may get lost as other optimization like CSE or
    constant folding may canonicalize them away.

    To address the problem, for each *use* of the plaintext, we insert an `mgmt.init` operation
    to initialize the plaintext with `mgmt` attributes.

    Technical reasons for registering memory effects:
    Register a (bogus) memory effect to prevent CSE from merging this op. Two mgmt.init ops
    could be seen as equivalent only if they have the same MgmtAttr with *level/dimension/scale*
    annotated, otherwise we could not judge whether they are equivalent or not. In practice,
    we create the op first and only in later analyses we know whether they are equivalent or not.

    ConditionallySpeculatable is for isSpeculatable check in hoisting canonicalization.

    Syntax: mgmt.init operands attr-dict : type($output)
    Example: %result = mgmt.init %input : i32
    """

    name = "mgmt.init"

    T: ClassVar = VarConstraint("T", BaseAttr(Attribute))

    input = operand_def(T)
    output = result_def(T)

    # Optional management attributes for level, dimension, scale, etc.
    level = opt_attr_def(IntegerAttr)
    dimension = opt_attr_def(IntegerAttr)
    scale = opt_attr_def(IntegerAttr)
    mgmt_type = opt_attr_def(StringAttr, attr_name="mgmt_type")

    traits = traits_def(SameOperandsAndResultType())

    # Custom parser and printer to support: mgmt.init %input : type

    assembly_format = "$input attr-dict `:` type($input)"


MGMT = Dialect(
    "mgmt",
    [
        InitOp,
    ],
    [
        # Attributes would go here if we define custom mgmt attribute types
    ],
)
