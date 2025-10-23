"""
Dataset-specific derivation modules for the PBUF data preparation framework.

This package contains derivation modules for each supported dataset type:
- sn_derivation: Supernova data processing
- bao_derivation: BAO isotropic/anisotropic processing
- cmb_derivation: CMB distance priors processing
- cc_derivation: Cosmic Chronometers processing
- rsd_derivation: RSD growth rate processing

Each module implements the DerivationModule interface to provide standardized
transformation logic while maintaining dataset-specific processing requirements.
"""

# Import modules conditionally to handle missing dependencies gracefully
__all__ = []

try:
    from .sn_derivation import SNDerivationModule
    __all__.append('SNDerivationModule')
except ImportError as e:
    print(f"Warning: Could not import SNDerivationModule: {e}")
    SNDerivationModule = None

try:
    from .bao_derivation import BAODerivationModule
    __all__.append('BAODerivationModule')
except ImportError as e:
    print(f"Warning: Could not import BAODerivationModule: {e}")
    BAODerivationModule = None

try:
    from .cmb_derivation import CMBDerivationModule
    __all__.append('CMBDerivationModule')
except ImportError as e:
    print(f"Warning: Could not import CMBDerivationModule: {e}")
    CMBDerivationModule = None

try:
    from .cc_derivation import CCDerivationModule
    __all__.append('CCDerivationModule')
except ImportError as e:
    print(f"Warning: Could not import CCDerivationModule: {e}")
    CCDerivationModule = None

try:
    from .rsd_derivation import RSDDerivationModule
    __all__.append('RSDDerivationModule')
except ImportError as e:
    print(f"Warning: Could not import RSDDerivationModule: {e}")
    RSDDerivationModule = None