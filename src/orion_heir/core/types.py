"""
Core types and data structures for the Orion-HEIR translator.

This module contains the fundamental data structures used throughout
the translator, avoiding circular import issues.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Protocol
from dataclasses import dataclass


@dataclass
class FHEOperation:
    """
    Generic representation of an FHE operation.
    
    This is frontend-agnostic and can represent operations from any FHE library.
    """
    op_type: str  # e.g., "add", "mul", "rotate"
    method_name: str  # Original method name from frontend
    args: List[Any]
    kwargs: Dict[str, Any]
    result_var: Optional[str] = None
    level: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class SchemeParameters(Protocol):
    """Protocol for FHE scheme parameters."""
    
    @property
    def ring_degree(self) -> int:
        """Ring degree of the polynomial."""
        ...
    
    @property
    def ciphertext_modulus_chain(self) -> List[int]:
        """Chain of moduli for ciphertexts."""
        ...
    
    @property
    def plaintext_modulus(self) -> int:
        """Modulus for plaintexts."""
        ...


class FrontendInterface(ABC):
    """Abstract interface for FHE library frontends."""
    
    @abstractmethod
    def extract_operations(self, source: Any) -> List[FHEOperation]:
        """Extract FHE operations from the source representation."""
        pass
    
    @abstractmethod
    def extract_scheme_parameters(self, source: Any) -> SchemeParameters:
        """Extract scheme parameters from the source."""
        pass
