"""
Orion scheme parameters implementation.

This module provides Orion-specific scheme parameter handling,
including integration with the actual Orion library when available.
"""

from typing import List, Optional, Union
import warnings

from ...core.types import SchemeParameters


class OrionNotAvailableError(Exception):
    """Raised when Orion is required but not available."""
    pass


class OrionSchemeParameters(SchemeParameters):
    """
    Orion-specific implementation of scheme parameters.
    
    This class handles Orion CKKS parameters and provides
    the interface expected by the generic translator.
    """
    
    def __init__(self, 
                 logN: Union[List[int], int],
                 logQ: List[int], 
                 logP: List[int],
                 logScale: int,
                 slots: int,
                 ring_degree: int,
                 backend: str = "lattigo",
                 require_orion: bool = False):
        # Normalize logN to always be an integer for Orion
        self.logN = logN[0] if isinstance(logN, list) else logN
        self.logQ = logQ
        self.logP = logP
        self.logScale = logScale
        self.slots = slots
        self.backend = backend
        self._ring_degree = ring_degree
        self._modulus_chain = None
        self.require_orion = require_orion
        
        # Try to get actual primes from Orion if available
        self._actual_primes = self._get_actual_primes()
    
    @property
    def ring_degree(self) -> int:
        """Ring degree of the polynomial."""
        return self._ring_degree
    
    @property
    def ciphertext_modulus_chain(self) -> List[int]:
        """Chain of moduli for ciphertexts."""
        if self._actual_primes:
            return self._actual_primes
        
        if self.require_orion:
            raise OrionNotAvailableError(
                "Orion FHE library is required but not available. "
                "Please install orion-fhe or set require_orion=False to use fallback primes."
            )
        
        # Fallback to computed primes
        return self._compute_fallback_primes()
    
    @property
    def plaintext_modulus(self) -> int:
        """Modulus for plaintexts."""
        # For CKKS, this is typically derived from the scaling factor
        return 2 ** self.logScale
    
    @property
    def log_scale(self) -> int:
        """Logarithm of the scaling factor."""
        return self.logScale
    
    @property
    def log_n(self) -> int:
        """Logarithm of the ring degree."""
        return self.logN
    
    def _get_actual_primes(self) -> Optional[List[int]]:
        """
        Get actual primes from Orion if available.
        
        This method attempts to create an Orion scheme and extract
        the actual moduli used. Falls back gracefully if Orion
        is not available.
        """
        try:
            from orion.core.orion import Scheme
            
            # Create Orion config - ensure LogN is an integer, not a list
            config = {
                'ckks_params': {
                    'LogN': self.logN,  # Now guaranteed to be an integer
                    'LogQ': self.logQ,
                    'LogP': self.logP,
                    'LogScale': self.logScale,
                    'H': 192,
                    'RingType': 'standard'
                },
                'orion': {
                    'margin': 2,
                    'embedding_method': 'hybrid',
                    'backend': self.backend,
                    'fuse_modules': True,
                    'debug': False,
                    'diags_path': '',
                    'keys_path': '',
                    'io_mode': 'none'
                }
            }
            
            print(f"🔧 Initializing Orion with LogN={self.logN}, LogQ={self.logQ}")
            
            # Initialize scheme
            scheme = Scheme()
            scheme.init_scheme(config)
            
            # Get moduli
            if hasattr(scheme, 'encoder') and scheme.encoder is not None:
                moduli = scheme.encoder.get_moduli_chain()
                print(f"📊 Retrieved actual Orion moduli: {moduli}")
                
                # Clean up
                scheme.delete_scheme()
                return moduli
            else:
                if self.require_orion:
                    raise OrionNotAvailableError("Orion encoder not initialized properly")
                warnings.warn("Orion encoder not initialized, using fallback primes")
                
        except ImportError as e:
            if self.require_orion:
                raise OrionNotAvailableError(
                    f"Orion FHE library is required but not installed. "
                    f"Install with: pip install orion-fhe\n"
                    f"Original error: {e}"
                )
            print("⚠️ Orion not available, using computed fallback primes")
        except Exception as e:
            if self.require_orion:
                raise OrionNotAvailableError(f"Error initializing Orion: {e}")
            warnings.warn(f"Error getting Orion primes: {e}, using fallback")
        
        return None
    
    def _compute_fallback_primes(self) -> List[int]:
        """
        Compute fallback primes when Orion is not available.
        
        This generates reasonable prime values based on the
        logarithmic specifications.
        """
        primes = []
        
        # Generate primes based on logQ values
        for log_q in self.logQ:
            # Use a simple method to find a prime near 2^log_q
            target = 2 ** log_q
            prime = self._find_prime_near(target)
            primes.append(prime)
        
        print(f"📊 Using computed fallback primes: {primes}")
        return primes
    
    def _find_prime_near(self, target: int) -> int:
        """
        Find a prime number near the target value.
        
        This is a simple implementation for fallback purposes.
        """
        # Start from target and work downwards
        candidate = target - 1
        if candidate % 2 == 0:
            candidate -= 1
        
        while candidate > target // 2:
            if self._is_prime(candidate):
                return candidate
            candidate -= 2
        
        # Fallback to target if no prime found
        return target
    
    def _is_prime(self, n: int) -> bool:
        """Simple primality test for small numbers."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def to_dict(self) -> dict:
        """Convert parameters to dictionary representation."""
        return {
            'logN': self.logN,
            'logQ': self.logQ,
            'logP': self.logP,
            'logScale': self.logScale,
            'slots': self.slots,
            'ring_degree': self.ring_degree,
            'backend': self.backend,
            'require_orion': self.require_orion,
            'actual_primes': self.ciphertext_modulus_chain
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'OrionSchemeParameters':
        """Create parameters from dictionary representation."""
        return cls(
            logN=data['logN'],
            logQ=data['logQ'],
            logP=data['logP'],
            logScale=data['logScale'],
            slots=data['slots'],
            ring_degree=data['ring_degree'],
            backend=data.get('backend', 'lattigo'),
            require_orion=data.get('require_orion', False)
        )
    
    def __repr__(self) -> str:
        return (f"OrionSchemeParameters(logN={self.logN}, logQ={self.logQ}, "
                f"logP={self.logP}, logScale={self.logScale}, "
                f"slots={self.slots}, ring_degree={self.ring_degree}, "
                f"require_orion={self.require_orion})")


def create_default_orion_parameters(require_orion: bool = False) -> OrionSchemeParameters:
    """Create default Orion parameters for testing."""
    return OrionSchemeParameters(
        logN=13,  # Changed from [13] to 13
        logQ=[55, 45, 45, 55],
        logP=[55],
        logScale=45,
        slots=4096,
        ring_degree=8192,
        backend='lattigo',
        require_orion=require_orion
    )


def create_mlp_orion_parameters(require_orion: bool = False) -> OrionSchemeParameters:
    """Create Orion parameters suitable for MLP computations."""
    return OrionSchemeParameters(
        logN=13,  # Changed from [13] to 13
        logQ=[60, 50, 50, 60],
        logP=[60],
        logScale=50,
        slots=4096,
        ring_degree=8192,
        backend='lattigo',
        require_orion=require_orion
    )


def create_orion_parameters_strict() -> OrionSchemeParameters:
    """Create Orion parameters that require the actual Orion library."""
    return OrionSchemeParameters(
        logN=13,  # Changed from [13] to 13
        logQ=[55, 45, 45, 55],
        logP=[55],
        logScale=45,
        slots=4096,
        ring_degree=8192,
        backend='lattigo',
        require_orion=True  # This will hard fail if Orion is not available
    )
