"""
Orion-HEIR Translator Package

A standalone translator for converting Orion FHE operations to HEIR MLIR format.
"""

__version__ = "0.1.0"
__author__ = "FHE Research Team"

from .core.translator import GenericTranslator, create_translator
from .core.types import FHEOperation
from .frontends.orion.orion_frontend import OrionFrontend
from .frontends.orion.scheme_params import OrionSchemeParameters

__all__ = [
    'GenericTranslator',
    'FHEOperation', 
    'create_translator',
    'OrionFrontend',
    'OrionSchemeParameters',
]
