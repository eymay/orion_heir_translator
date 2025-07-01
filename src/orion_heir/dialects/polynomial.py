"""
Polynomial dialect implementation for xDSL
Provides polynomial ring attributes (simplified to only include what's needed)
"""

from xdsl.dialects.transform import Sequence
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
)
from xdsl.parser import Parser
from xdsl.printer import Printer


@irdl_attr_definition
class PolynomialAttr(ParametrizedAttribute):
    """
    Represents a polynomial as an attribute.

    For simplicity, we represent this as a string.

    Example: <1 + x**1024>
    """

    name = "polynomial.polynomial"

    polynomial_str: ParameterDef[Attribute]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """
        Parse polynomial string representation.
        
        We need to carefully parse everything between < and > as the polynomial expression.
        """
        parser.parse_punctuation("<")
        
        # Collect all tokens until we hit the closing >
        # This handles complex expressions like "1 + x**16"
        polynomial_tokens = []
        bracket_depth = 0
        
        while True:
            token = parser._current_token
            
            if token.text == ">" and bracket_depth == 0:
                # This is our closing bracket
                break
            elif token.text == ">":
                bracket_depth -= 1
            elif token.text == "<":
                bracket_depth += 1
            
            polynomial_tokens.append(token.text)
            parser._consume_token()
        
        # Join tokens with appropriate spacing
        polynomial_expr = ""
        for i, token in enumerate(polynomial_tokens):
            polynomial_expr += token
            
            # Add space between tokens (except around certain operators)
            if (i < len(polynomial_tokens) - 1 and 
                token not in "+-*^()" and 
                polynomial_tokens[i + 1] not in "+-*^()"):
                polynomial_expr += " "
        
        parser.parse_punctuation(">")
        
        polynomial_str_attr = StringAttr(polynomial_expr.strip())
        return [polynomial_str_attr]

    def print_parameters(self, printer: Printer) -> None:
        """Print polynomial string representation."""
        printer.print_string("<")
        printer.print_string(self.polynomial_str.data)
        printer.print_string(">")


@irdl_attr_definition
class RingAttr(ParametrizedAttribute):

    name = "polynomial.ring"

    coefficient_type: ParameterDef[Attribute]  # This can store types as well
    polynomial_modulus: ParameterDef[PolynomialAttr]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse ring coefficient type and polynomial modulus."""
        parser.parse_punctuation("<")
        
        parser.parse_keyword("coefficientType")
        parser.parse_punctuation("=")
        coefficient_type = parser.parse_attribute()
        
        parser.parse_punctuation(",")
        
        parser.parse_keyword("polynomialModulus")
        parser.parse_punctuation("=")
        
        polynomial_modulus_params = PolynomialAttr.parse_parameters(parser)
        polynomial_modulus = PolynomialAttr.new(polynomial_modulus_params)
        
        parser.parse_punctuation(">")
        
        return [coefficient_type, polynomial_modulus]

    def print_parameters(self, printer: Printer) -> None:
        """Print ring coefficient type and polynomial modulus."""
        printer.print_string("<coefficientType = ")
        printer.print_attribute(self.coefficient_type)
        printer.print_string(", polynomialModulus = ")
        self.polynomial_modulus.print_parameters(printer)
        printer.print_string(">")

    def get_alias_suffix(self, os):
        """Helper method for generating type aliases."""
        # This method is referenced in your LWE attributes
        # Add coefficient type info
        if hasattr(self.coefficient_type, 'width'):
            os.write(f"_{self.coefficient_type.__class__.__name__.lower()}")
        else:
            os.write(f"_{str(self.coefficient_type).replace('.', '_')}")
            
        # Add polynomial info (simplified)
        poly_str = self.polynomial_modulus.polynomial_str.data
        if "x**" in poly_str:
            # Extract degree from expressions like "1 + x**16"
            import re
            match = re.search(r'x\*\*(\d+)', poly_str)
            if match:
                degree = match.group(1)
                os.write(f"_x{degree}")
        else:
            os.write("_poly")



Polynomial = Dialect(
    "polynomial",
    [
        # No operations needed
    ],
    [
        PolynomialAttr,
        RingAttr,
    ],
)
