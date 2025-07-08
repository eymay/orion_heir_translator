"""
LWE (Learning With Errors) dialect implementation for xDSL
Provides LWE cryptographic types and operations
"""

from collections.abc import Sequence

from xdsl.dialects.builtin import (
    IntAttr,
    IntegerAttr,
    IntegerType,
    ArrayAttr,
    StringAttr,
    TensorType,
)
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    ParameterDef,
    irdl_attr_definition,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    result_def,
    attr_def,
    traits_def,
)
from xdsl.parser import Parser
from .declarative_parser import parse_parameters_declarative, create_field_specs, FieldType, FieldSpec
from xdsl.printer import Printer
from xdsl.traits import Pure
from xdsl.utils.exceptions import ParseError
# Import our custom dialects
from .polynomial import RingAttr, PolynomialAttr


@irdl_attr_definition
class InverseCanonicalEncodingAttr(ParametrizedAttribute):
    name = "lwe.inverse_canonical_encoding"

    scaling_factor: ParameterDef[IntegerAttr]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """
        Parse the scaling_factor parameter from syntax like:
        #lwe.inverse_canonical_encoding<scaling_factor = 0>
        """
        parser.parse_punctuation("<")
        
        # Parse "scaling_factor"
        parser.parse_keyword("scaling_factor")
        
        # Parse "="
        parser.parse_punctuation("=")
        
        # Parse the integer value
        scaling_factor_value = parser.parse_integer()
        scaling_factor_attr = IntegerAttr.from_int_and_width(scaling_factor_value, 64)
        
        parser.parse_punctuation(">")
        
        return [scaling_factor_attr]

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<scaling_factor = ")
        printer.print_string(str(self.scaling_factor.value.data))
        printer.print_string(">")


@irdl_attr_definition
class FullCRTPackingEncodingAttr(ParametrizedAttribute):
    """
    An encoding attribute for full CRT packing.

    This encoding maps a list of integers via the Chinese Remainder Theorem (CRT)
    into the plaintext space. This attribute can only be used in the context of
    full CRT packing, where the polynomial f(x) splits completely (into linear
    factors) and the number of slots equals the degree of f(x). This happens
    when q is prime and q = 1 mod n.

    Syntax: #lwe.full_crt_packing_encoding<scaling_factor = value>
    Example: #lwe.full_crt_packing_encoding<scaling_factor = 0>
    """

    name = "lwe.full_crt_packing_encoding"

    scaling_factor: ParameterDef[IntegerAttr]


@irdl_attr_definition
class KeyAttr(ParametrizedAttribute):
    """
    An attribute describing cryptographic keys.

    Syntax: #lwe.key<>
    Example: #lwe.key<>
    """

    name = "lwe.key"

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        parser.parse_punctuation("<")
        parser.parse_punctuation(">")
        return []

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("#" + self.name + "<>")


@irdl_attr_definition
class ModulusChainAttr(ParametrizedAttribute):
    """
    An attribute describing modulus chains for RLWE schemes.

    Syntax: #lwe.modulus_chain<elements = <val1, val2, ...>, current = idx>
    Example: #lwe.modulus_chain<elements = <1095233372161 : i64, 1032955396097 : i64>, current = 0>
    """

    name = "lwe.modulus_chain"

    elements: ParameterDef[ArrayAttr[IntegerAttr]]
    current: ParameterDef[IntegerAttr]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse modulus chain parameters declaratively."""
        
        field_specs = create_field_specs(
            elements=(FieldType.ARRAY_OF_INTEGERS, True),
            current=(FieldType.INTEGER, True)
        )
        
        return parse_parameters_declarative(parser, field_specs)

    def print_parameters(self, printer: Printer) -> None:
        """Print in HEIR-compatible format with square brackets."""
        printer.print_string("<elements = <")
        for i, element in enumerate(self.elements.data):
            if i > 0:
                printer.print_string(", ")
            printer.print_string(str(element.value.data))
            printer.print_string(" : i64")
        printer.print_string(">, current = ")
        printer.print_string(str(self.current.value.data))
        printer.print_string(">")

@irdl_attr_definition
class PlaintextSpaceAttr(ParametrizedAttribute):
    """
    An attribute describing the plaintext space.

    Syntax: #lwe.plaintext_space<ring = ring_attr, encoding = encoding_attr>
    Example: #lwe.plaintext_space<ring = #ring_Z65537_i64_1_x1024_, encoding = #inverse_canonical_encoding>
    """

    name = "lwe.plaintext_space"

    ring: ParameterDef[RingAttr]
    encoding: ParameterDef[Attribute]  # Could be various encoding types

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse ring and encoding parameters declaratively."""
        
        # Import here to avoid circular imports
        from .polynomial import RingAttr
        
        field_specs = create_field_specs(
            ring=(FieldType.FLEXIBLE_ATTRIBUTE, True, RingAttr),
            encoding=(FieldType.ATTRIBUTE, True)
        )
        
        return parse_parameters_declarative(parser, field_specs)




    def print_parameters(self, printer: Printer) -> None:
        """Print ring and encoding parameters."""
        printer.print_string("<ring = ")
        self.ring.print_parameters(printer)
        printer.print_string(", encoding = ")
        printer.print_attribute(self.encoding)
        printer.print_string(">")



@irdl_attr_definition
class CiphertextSpaceAttr(ParametrizedAttribute):
    """
    An attribute describing the ciphertext space.

    Syntax: #lwe.ciphertext_space<ring = ring_attr, encryption_type = type, size = value>
    Example: #lwe.ciphertext_space<ring = #ring_rns_L1_1_x16_, encryption_type = lsb, size = 3>
    """

    name = "lwe.ciphertext_space"

    ring: ParameterDef[RingAttr]
    encryption_type: ParameterDef[StringAttr]  # "lsb", "msb", etc.
    size: ParameterDef[IntegerAttr]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse ring, encryption_type, and size parameters declaratively."""
        
        from .polynomial import RingAttr
        
        field_specs = create_field_specs(
            ring=(FieldType.FLEXIBLE_ATTRIBUTE, True, RingAttr),
            encryption_type=(FieldType.IDENTIFIER, True),
            size=(FieldType.INTEGER, False, 2)  # Optional with default value 2
        )
        
        return parse_parameters_declarative(parser, field_specs)



    def get_alias_suffix(self, os) -> None:
        """Helper method for generating type aliases."""
        self.ring.get_alias_suffix(os)
        if self.size.data != 2:
            os.write(f"_D{self.size.value}")

    def print_parameters(self, printer: Printer) -> None:
        """Print ring, encryption_type, and size parameters."""
        printer.print_string("<ring = ")
        printer.print_attribute(self.ring)
        printer.print_string(", encryption_type = ")
        printer.print_string(self.encryption_type.data)
        printer.print_string(", size = ")
        printer.print_string(str(self.size.value.data))
        printer.print_string(">")


@irdl_attr_definition
class ApplicationDataAttr(ParametrizedAttribute):
    """
    An attribute describing application data semantics.

    Syntax: #lwe.application_data<message_type = type>
    Example: <message_type = i3>
    """

    name = "lwe.application_data"

    message_type: ParameterDef[Attribute]

    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse message_type parameter."""
        parser.parse_punctuation("<")
        parser.parse_keyword("message_type")
        parser.parse_punctuation("=")
        message_type = parser.parse_type()
        parser.parse_punctuation(">")
        return [message_type]

    def print_parameters(self, printer: Printer) -> None:
        """Print message_type parameter."""
        printer.print_string("<message_type = ")
        printer.print_attribute(self.message_type)
        printer.print_string(">")

@irdl_attr_definition
class NewLWEPlaintextType(ParametrizedAttribute, TypeAttribute):
    """
    A type representing LWE plaintexts.

    Syntax: !lwe.new_lwe_plaintext<application_data = app_data, plaintext_space = space>
    Example: !lwe.new_lwe_plaintext<application_data = <message_type = i3>, plaintext_space = #plaintext_space>
    """

    name = "lwe.new_lwe_plaintext"

    application_data: ParameterDef[ApplicationDataAttr]
    plaintext_space: ParameterDef[PlaintextSpaceAttr]

    @classmethod  
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse plaintext type parameters declaratively."""
        
        field_specs = {
            "application_data": FieldSpec(
                "application_data",
                FieldType.NESTED_PARAMETERS,
                required=True,
                nested_parser=ApplicationDataAttr.parse_parameters,
                attribute_class=ApplicationDataAttr
            ),
            "plaintext_space": FieldSpec(
                "plaintext_space", 
                FieldType.NESTED_PARAMETERS,
                required=True,
                nested_parser=PlaintextSpaceAttr.parse_parameters,
                attribute_class=PlaintextSpaceAttr
            )
        }
        
        return parse_parameters_declarative(parser, field_specs)

        def print_parameters(self, printer: Printer) -> None:
            """Print application_data and plaintext_space parameters."""
            printer.print_string("<application_data = ")
            self.application_data.print_parameters(printer)
            printer.print_string(", plaintext_space = ")
            self.plaintext_space.print_parameters(printer)
            printer.print_string(">")


@irdl_attr_definition
class NewLWECiphertextType(ParametrizedAttribute, TypeAttribute):
    """
    A type representing LWE ciphertexts.

    Syntax: !lwe.new_lwe_ciphertext<application_data = app_data, plaintext_space = space,
                                   ciphertext_space = c_space, key = key_attr, modulus_chain = chain>
    """

    name = "lwe.new_lwe_ciphertext"

    application_data: ParameterDef[ApplicationDataAttr]
    plaintext_space: ParameterDef[PlaintextSpaceAttr]
    ciphertext_space: ParameterDef[CiphertextSpaceAttr]
    key: ParameterDef[KeyAttr]
    modulus_chain: ParameterDef[ModulusChainAttr]


    @classmethod
    def parse_parameters(cls, parser: Parser) -> Sequence[Attribute]:
        """Parse all ciphertext type parameters declaratively."""
        
        field_specs = {
            "application_data": FieldSpec(
                "application_data", 
                FieldType.NESTED_PARAMETERS,
                required=True,
                nested_parser=ApplicationDataAttr.parse_parameters,
                attribute_class=ApplicationDataAttr
            ),
            "plaintext_space": FieldSpec(
                "plaintext_space",
                FieldType.NESTED_PARAMETERS, 
                required=True,
                nested_parser=PlaintextSpaceAttr.parse_parameters,
                attribute_class=PlaintextSpaceAttr
            ),
            "ciphertext_space": FieldSpec(
                "ciphertext_space",
                FieldType.NESTED_PARAMETERS,
                required=True, 
                nested_parser=CiphertextSpaceAttr.parse_parameters,
                attribute_class=CiphertextSpaceAttr
            ),
            "key": FieldSpec(
                "key",
                FieldType.ATTRIBUTE,
                required=True
            ),
            "modulus_chain": FieldSpec(
                "modulus_chain",
                FieldType.FLEXIBLE_ATTRIBUTE,
                required=False,
                attribute_class=ModulusChainAttr
            )
        }
        
        result = parse_parameters_declarative(parser, field_specs)
        
        # If modulus_chain wasn't provided, create a default one
        if len(result) == 4:  # No modulus_chain
            empty_elements = ArrayAttr([])
            default_current = IntegerAttr.from_int_and_width(0, 64) 
            default_modulus_chain = ModulusChainAttr([empty_elements, default_current])
            result.append(default_modulus_chain)
        
        return result



    def print_parameters(self, printer: Printer) -> None:
        """Print all ciphertext type parameters."""
        printer.print_string("<application_data = ")
        self.application_data.print_parameters(printer)
        printer.print_string(", plaintext_space = ")
        self.plaintext_space.print_parameters(printer)
        printer.print_string(", ciphertext_space = ")
        self.ciphertext_space.print_parameters(printer)
        printer.print_string(", key = ")
        self.key.print_parameters(printer)
        
        # Only print modulus_chain if it's not empty/default
        if (hasattr(self.modulus_chain, 'elements') and 
            len(self.modulus_chain.elements.data) > 0):
            printer.print_string(", modulus_chain = ")
            self.modulus_chain.print_parameters(printer)
        
        printer.print_string(">")


@irdl_op_definition
class RLWEEncodeOp(IRDLOperation):
    """
    Encode an integer/tensor to yield an RLWE plaintext.

    This op uses an encoding attribute to encode the bits of the input into an RLWE
    plaintext value that can then be encrypted. CKKS cleartext inputs may be floating
    points, and a scaling factor described by the encoding will be applied.

    Example: %pt_62 = lwe.rlwe_encode %cst_0 {encoding = #inverse_canonical_encoding, ring = #ring_f64_1_x16} : tensor<1x16xf32> -> !pt
    """

    name = "lwe.rlwe_encode"

    # Operands - allow tensors, integers, or floats
    input = operand_def(TensorType)
    output = result_def(NewLWEPlaintextType)

    # Attributes
    encoding = attr_def(InverseCanonicalEncodingAttr)
    ring = attr_def(RingAttr)  # Polynomial ring attribute

    traits = traits_def(Pure())

    # Remove custom assembly format and use declarative format instead
    assembly_format = "$input attr-dict `:` type($input) `->` type($output)"


LWE = Dialect(
    "lwe",
    [
        RLWEEncodeOp,
    ],
    [
        InverseCanonicalEncodingAttr,
        FullCRTPackingEncodingAttr,
        KeyAttr,
        ModulusChainAttr,
        PlaintextSpaceAttr,
        CiphertextSpaceAttr,
        ApplicationDataAttr,
        NewLWEPlaintextType,
        NewLWECiphertextType,
    ],
)
