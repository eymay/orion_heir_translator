"""
RNS (Residue Number System) dialect implementation for xDSL
Provides types and operations for RNS representations
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import ArrayAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class RNSType(ParametrizedAttribute, TypeAttribute):
    """
    A type representing integers in Residue Number System (RNS) form.

    The RNS representation stores an integer as a vector of residues modulo
    different primes, allowing for efficient parallel arithmetic operations.

    Syntax: !rns.rns<type1, type2, ...>
    Example: !rns.rns<!mod_arith.int<1095233372161 : i64>, !mod_arith.int<1032955396097 : i64>>
    """

    name = "rns.rns"

    basis_types: ParameterDef[ArrayAttr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<")
        for i, basis_type in enumerate(self.basis_types.data):
            if i > 0:
                printer.print_string(", ")
            printer.print_attribute(basis_type)
        printer.print_string(">")

    @classmethod
    def parse_parameters(cls, parser: Parser) -> list[Attribute]:
        parser.parse_punctuation("<")

        basis_types = []

        # Parse first type
        basis_types.append(parser.parse_type())

        # Parse remaining types
        while parser.parse_optional_punctuation(","):
            basis_types.append(parser.parse_type())

        parser.parse_punctuation(">")

        # Convert to ArrayAttr
        basis_types_attr = ArrayAttr(basis_types)

        return [basis_types_attr]

    def verify_(self) -> None:
        """Verify that all basis types are valid modular arithmetic types."""
        if not self.basis_types.data:
            raise ValueError("RNS type must have at least one basis type")

        # Additional verification could check that basis types are mod_arith types
        # and that their moduli are pairwise coprime, but we'll keep it simple for now


RNS = Dialect(
    "rns",
    [
        # RNS dialect currently only defines types, no operations
        # Operations would be added here if needed
    ],
    [
        RNSType,
    ],
)
