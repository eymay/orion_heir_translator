"""
Declarative parser helper for HEIR attributes.

This helper takes a specification of expected fields and handles all parsing variations.

Add this to: src/orion_heir/dialects/declarative_parser.py
"""

from typing import Dict, Any, Sequence, Union, Optional, Type, Callable
from enum import Enum
from dataclasses import dataclass
from xdsl.ir import Attribute
from xdsl.parser import Parser
from xdsl.dialects.builtin import StringAttr, IntegerAttr, IntegerType, ArrayAttr


class FieldType(Enum):
    """Types of fields that can be parsed."""
    TYPE = "type"                    # parser.parse_type()
    ATTRIBUTE = "attribute"          # parser.parse_attribute()
    INTEGER = "integer"              # parser.parse_integer()
    IDENTIFIER = "identifier"        # parser.parse_identifier()
    STRING = "string"                # parser.parse_string()
    FLEXIBLE_ATTRIBUTE = "flexible_attribute"  # Can be inline or referenced
    ARRAY_OF_INTEGERS = "array_of_integers"    # [val1 : type, val2 : type, ...]
    NESTED_PARAMETERS = "nested_parameters"    # Recursively parse another attribute's parameters


@dataclass
class FieldSpec:
    """Specification for a single field."""
    name: str
    field_type: FieldType
    required: bool = True
    default_value: Any = None
    # For flexible attributes - the attribute class to create
    attribute_class: Optional[Type] = None
    # For nested parameters - the parsing function to call
    nested_parser: Optional[Callable] = None
    # For arrays - the element type
    element_type: Optional[str] = None


def parse_parameters_declarative(parser: Parser, field_specs: Dict[str, FieldSpec]) -> Sequence[Attribute]:
    """
    Parse parameters declaratively based on field specifications.
    
    Args:
        parser: xDSL parser
        field_specs: Dictionary mapping field names to their specifications
        
    Returns:
        List of parsed attributes in the order they appear in field_specs
        
    Example:
        field_specs = {
            "ring": FieldSpec("ring", FieldType.FLEXIBLE_ATTRIBUTE, attribute_class=RingAttr),
            "encoding": FieldSpec("encoding", FieldType.ATTRIBUTE),
            "size": FieldSpec("size", FieldType.INTEGER, required=False, default_value=2)
        }
    """
    parser.parse_punctuation("<")
    
    parsed_values = {}
    
    # Parse all fields until we hit the closing >
    while parser._current_token.text != ">":
        # Parse field name
        field_name = parser.parse_identifier()
        
        if field_name not in field_specs:
            raise Exception(f"Unexpected field: {field_name}")
        
        spec = field_specs[field_name]
        
        # Parse the = separator
        parser.parse_punctuation("=")
        
        # Parse the value based on the field type
        value = _parse_field_value(parser, spec)
        parsed_values[field_name] = value
        
        # Check for comma (more fields) or > (end)
        if parser._current_token.text == ",":
            parser.parse_punctuation(",")
        elif parser._current_token.text == ">":
            break
        else:
            raise Exception(f"Expected ',' or '>' after field {field_name}")
    
    parser.parse_punctuation(">")
    
    # Build the result list in the order specified by field_specs
    result = []
    for field_name, spec in field_specs.items():
        if field_name in parsed_values:
            result.append(parsed_values[field_name])
        elif not spec.required:
            # Use default value
            default = _create_default_value(spec)
            result.append(default)
        else:
            raise Exception(f"Required field missing: {field_name}")
    
    return result


def _parse_field_value(parser: Parser, spec: FieldSpec) -> Any:
    """Parse a single field value based on its specification."""
    
    if spec.field_type == FieldType.TYPE:
        return parser.parse_type()
    
    elif spec.field_type == FieldType.ATTRIBUTE:
        return parser.parse_attribute()
    
    elif spec.field_type == FieldType.INTEGER:
        return IntegerAttr.from_int_and_width(parser.parse_integer(), 64)
    
    elif spec.field_type == FieldType.IDENTIFIER:
        return StringAttr(parser.parse_identifier())
    
    elif spec.field_type == FieldType.STRING:
        return StringAttr(parser.parse_string())
    
    elif spec.field_type == FieldType.FLEXIBLE_ATTRIBUTE:
        return _parse_flexible_attribute(parser, spec)
    
    elif spec.field_type == FieldType.ARRAY_OF_INTEGERS:
        return _parse_integer_array(parser, spec.element_type or "i64")
    
    elif spec.field_type == FieldType.NESTED_PARAMETERS:
        if spec.nested_parser is None:
            raise Exception(f"nested_parser required for NESTED_PARAMETERS field {spec.name}")
        params = spec.nested_parser(parser)
        if spec.attribute_class:
            return spec.attribute_class.new(params)
        else:
            return params
    
    else:
        raise Exception(f"Unknown field type: {spec.field_type}")


def _parse_flexible_attribute(parser: Parser, spec: FieldSpec) -> Attribute:
    """Parse an attribute - handles both referenced and inline formats."""
    current_token = parser._current_token.text
    
    if spec.attribute_class is None:
        # Must use standard attribute parsing
        return parser.parse_attribute()
    
    # Skip any attribute reference prefix (like "#polynomial.ring")
    if current_token != "<":
        # This handles "#polynomial.ring", "polynomial.ring", etc.
        parser._consume_token()
    
    # Now parse the parameters using the attribute's own parser
    params = spec.attribute_class.parse_parameters(parser)
    return spec.attribute_class.new(params)


def _parse_attribute_manually(parser: Parser, spec: FieldSpec) -> Attribute:
    """Manually parse #dialect.attr<...> when automatic parsing fails."""
    # Skip the # and parse components
    parser.parse_punctuation("#")
    dialect_name = parser.parse_identifier()
    parser.parse_punctuation(".")
    attr_name = parser.parse_identifier()
    
    # Now parse as inline using the attribute class
    if spec.attribute_class is None:
        raise Exception(f"attribute_class required for manual parsing of {dialect_name}.{attr_name}")
    
    params = spec.attribute_class.parse_parameters(parser)
    return spec.attribute_class.new(params)


def _parse_integer_array(parser: Parser, element_type: str) -> ArrayAttr:
    """Parse an array of integers like <val1 : i64, val2 : i64, ...>"""
    parser.parse_punctuation("<")
    
    integer_values = []
    
    # Parse first integer
    value = parser.parse_integer()
    parser.parse_punctuation(":")
    parser.parse_identifier()  # type like "i64"
    integer_values.append(IntegerAttr.from_int_and_width(value, 64))
    
    # Parse remaining integers
    while parser.parse_optional_punctuation(","):
        value = parser.parse_integer()
        parser.parse_punctuation(":")
        parser.parse_identifier()  # type
        integer_values.append(IntegerAttr.from_int_and_width(value, 64))
    
    parser.parse_punctuation(">")
    
    return ArrayAttr(integer_values)


def _create_default_value(spec: FieldSpec) -> Attribute:
    """Create a default value for an optional field."""
    if spec.default_value is not None:
        if spec.field_type == FieldType.INTEGER:
            return IntegerAttr.from_int_and_width(spec.default_value, 64)
        elif spec.field_type == FieldType.IDENTIFIER or spec.field_type == FieldType.STRING:
            return StringAttr(spec.default_value)
        else:
            return spec.default_value
    else:
        # Create a reasonable default based on type
        if spec.field_type == FieldType.INTEGER:
            return IntegerAttr.from_int_and_width(0, 64)
        elif spec.field_type == FieldType.IDENTIFIER or spec.field_type == FieldType.STRING:
            return StringAttr("")
        elif spec.field_type == FieldType.ARRAY_OF_INTEGERS:
            return ArrayAttr([])
        else:
            raise Exception(f"No default value available for field {spec.name} of type {spec.field_type}")


# Convenience function for creating common field specifications
def create_field_specs(**kwargs) -> Dict[str, FieldSpec]:
    """
    Create field specifications using a convenient syntax.
    
    Example:
        create_field_specs(
            ring=(FieldType.FLEXIBLE_ATTRIBUTE, True, RingAttr),
            encoding=(FieldType.ATTRIBUTE, True),
            size=(FieldType.INTEGER, False, 2)
        )
    """
    result = {}
    
    for name, spec_tuple in kwargs.items():
        if len(spec_tuple) == 2:
            field_type, required = spec_tuple
            spec = FieldSpec(name, field_type, required)
        elif len(spec_tuple) == 3:
            field_type, required, default_or_class = spec_tuple
            if field_type == FieldType.FLEXIBLE_ATTRIBUTE:
                spec = FieldSpec(name, field_type, required, attribute_class=default_or_class)
            else:
                spec = FieldSpec(name, field_type, required, default_value=default_or_class)
        else:
            raise ValueError(f"Invalid spec tuple for {name}: {spec_tuple}")
        
        result[name] = spec
    
    return result
