"""
LWE and CKKS verification traits ported from HEIR to xDSL
These traits provide verification constraints for homomorphic encryption operations
"""

from abc import ABC
from typing import Any

from xdsl.ir import Operation, OpResult, SSAValue
from xdsl.traits import OpTrait
from xdsl.utils.exceptions import VerifyException

from .lwe import NewLWECiphertextType, NewLWEPlaintextType
from .polynomial import RingAttr


class SameOperandsAndResultRings(OpTrait):
    """
    Trait that ensures all operands and results have the same polynomial rings.
    
    This verifies that all ciphertext operands and results share the same
    underlying polynomial ring structure, which is required for ring-based
    homomorphic encryption operations like CKKS and RLWE.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        rings: RingAttr | None = None
        
        def check_ring(ring: RingAttr) -> None:
            nonlocal rings
            if rings is None:
                rings = ring
            elif rings != ring:
                raise VerifyException(
                    f"Operation {op.name} requires all operands and results "
                    f"to have the same rings, but found {rings} and {ring}"
                )
        
        # Check result types
        for result in op.results:
            if isinstance(result.type, NewLWECiphertextType):
                check_ring(result.type.ciphertext_space.ring)
        
        # Check operand types  
        for operand in op.operands:
            if isinstance(operand.type, NewLWECiphertextType):
                check_ring(operand.type.ciphertext_space.ring)


class SameOperandsAndResultPlaintextTypes(OpTrait):
    """
    Trait that ensures all operands and results have the same plaintext types.
    
    This verifies that the plaintext spaces derived from ciphertext types
    and any explicit plaintext operands all match. This is crucial for
    operations that mix ciphertexts and plaintexts.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        plaintext_type: NewLWEPlaintextType | None = None
        
        def check_plaintext_type(pt: NewLWEPlaintextType) -> None:
            nonlocal plaintext_type
            if plaintext_type is None:
                plaintext_type = pt
            elif plaintext_type != pt:
                raise VerifyException(
                    f"Operation {op.name} requires all operands and results "
                    f"to have the same plaintextTypes, but found {plaintext_type} and {pt}"
                )
        
        def get_plaintext_from_ciphertext(ct: NewLWECiphertextType) -> NewLWEPlaintextType:
            # Extract plaintext type from ciphertext type
            return NewLWEPlaintextType([
                ct.application_data,
                ct.plaintext_space
                ])
        
        # Check result types
        for result in op.results:
            if isinstance(result.type, NewLWECiphertextType):
                check_plaintext_type(get_plaintext_from_ciphertext(result.type))
        
        # Check operand types
        for operand in op.operands:
            if isinstance(operand.type, NewLWECiphertextType):
                check_plaintext_type(get_plaintext_from_ciphertext(operand.type))
            elif isinstance(operand.type, NewLWEPlaintextType):
                check_plaintext_type(operand.type)


class AllCiphertextTypesMatch(OpTrait):
    """
    Trait that ensures all ciphertext types are identical.
    
    This is a stricter constraint than SameOperandsAndResultRings,
    requiring that all ciphertext operands and results have exactly
    the same complete ciphertext type structure.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        ciphertext_type: NewLWECiphertextType | None = None
        
        def check_ciphertext_type(ct: NewLWECiphertextType) -> None:
            nonlocal ciphertext_type
            if ciphertext_type is None:
                ciphertext_type = ct
            elif ciphertext_type != ct:
                raise VerifyException(
                    f"Operation {op.name} requires all ciphertexts to have "
                    f"the same ciphertextType, but found {ciphertext_type} and {ct}"
                )
        
        # Check result types
        for result in op.results:
            if isinstance(result.type, NewLWECiphertextType):
                check_ciphertext_type(result.type)
        
        # Check operand types
        for operand in op.operands:
            if isinstance(operand.type, NewLWECiphertextType):
                check_ciphertext_type(operand.type)


class IsCiphertextPlaintextOp(OpTrait):
    """
    Trait that verifies an operation takes exactly one ciphertext and one plaintext
    operand (in either order) and produces a ciphertext result.
    
    This is commonly used for operations like add_plain, sub_plain, mul_plain
    in homomorphic encryption schemes.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        if len(op.operands) != 2:
            raise VerifyException(
                f"Ciphertext-plaintext operation {op.name} requires exactly "
                f"two operands, but got {len(op.operands)}"
            )
        
        if len(op.results) != 1:
            raise VerifyException(
                f"Ciphertext-plaintext operation {op.name} requires exactly "
                f"one result, but got {len(op.results)}"
            )
        
        operand_types = [operand.type for operand in op.operands]
        
        # Check that we have one ciphertext and one plaintext operand
        ciphertext_count = sum(1 for t in operand_types if isinstance(t, NewLWECiphertextType))
        plaintext_count = sum(1 for t in operand_types if isinstance(t, NewLWEPlaintextType))
        
        if ciphertext_count != 1 or plaintext_count != 1:
            raise VerifyException(
                f"Operation {op.name} expected exactly one ciphertext and one plaintext "
                f"operand, but got {ciphertext_count} ciphertext(s) and {plaintext_count} plaintext(s). "
                f"Operand types: {operand_types}"
            )
        
        # Check that result is a ciphertext
        result_type = op.results[0].type
        if not isinstance(result_type, NewLWECiphertextType):
            raise VerifyException(
                f"Operation {op.name} expected result to be ciphertext, "
                f"but got {result_type}"
            )


# Additional helper traits that might be useful

class SameCiphertextRings(OpTrait):
    """
    Simplified trait that only checks ciphertext operands have the same rings.
    Useful for operations that only work on ciphertexts.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        rings: RingAttr | None = None
        
        for operand in op.operands:
            if isinstance(operand.type, NewLWECiphertextType):
                operand_ring = operand.type.ciphertext_space.ring
                if rings is None:
                    rings = operand_ring
                elif rings != operand_ring:
                    raise VerifyException(
                        f"Operation {op.name} requires all ciphertext operands "
                        f"to have the same rings"
                    )


class SameApplicationData(OpTrait):
    """
    Trait that ensures all LWE types have the same application data.
    This can be useful for ensuring encoding compatibility.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        application_data: Any = None
        
        def check_application_data(app_data: Any) -> None:
            nonlocal application_data
            if application_data is None:
                application_data = app_data
            elif application_data != app_data:
                raise VerifyException(
                    f"Operation {op.name} requires all operands and results "
                    f"to have the same application data"
                )
        
        # Check all operands and results
        for value in list(op.operands) + list(op.results):
            if isinstance(value.type, NewLWECiphertextType):
                check_application_data(value.type.application_data)
            elif isinstance(value.type, NewLWEPlaintextType):
                check_application_data(value.type.application_data)

class AllTypesMatch(OpTrait):
    """
    Trait that ensures input and output types are exactly the same.
    Used for operations like negate, rotate where type is preserved.
    """
    
    @classmethod
    def verify(cls, op: Operation) -> None:
        if not op.operands or not op.results:
            return
        
        input_type = op.operands[0].type
        output_type = op.results[0].type
        
        if input_type != output_type:
            raise VerifyException(
                f"Operation {op.name} requires input and output types to match, "
                f"but got {input_type} and {output_type}"
            )

