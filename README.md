# PBUF Cosmology Pipeline

A unified cosmological parameter fitting infrastructure for PBUF and ΛCDM models.

## Quick Setup

### 1. Automatic Setup (Recommended)

Run the setup script to create the virtual environment and install dependencies:

```bash
./setup_env.sh
```

### 2. Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install core dependencies
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0
```

### 3. Verify Setup

Test that everything is working:

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Run setup test
python3 test_setup.py
```

## Project Structure

```
pipelines/
├── __init__.py                 # Main package initialization
├── fit_core/                   # Core fitting infrastructure
│   ├── __init__.py            # Core module with type definitions
│   ├── engine.py              # Unified optimization engine
│   ├── parameter.py           # Centralized parameter management
│   ├── likelihoods.py         # Block likelihood functions
│   ├── datasets.py            # Unified dataset loading
│   ├── statistics.py          # Statistical computations
│   ├── logging_utils.py       # Standardized logging
│   └── integrity.py           # Physics validation
├── fit_cmb.py                 # CMB fitting wrapper
├── fit_bao.py                 # BAO fitting wrapper
├── fit_aniso.py               # Anisotropic BAO wrapper
├── fit_sn.py                  # Supernova fitting wrapper
└── fit_joint.py               # Joint fitting wrapper
```

## Core Features

- **Unified Architecture**: Single optimization engine used by all fitters
- **Centralized Parameters**: Consistent parameter handling across all models
- **Physics Validation**: Comprehensive integrity checks and consistency tests
- **Standardized Logging**: Consistent diagnostic output and result reporting
- **Type Safety**: Full type hints for better development experience

## Usage

The pipeline is currently in development. Core interfaces and scaffolding are complete.

To test the basic functionality:

```python
import sys
sys.path.append('pipelines')

from fit_core import ParameterDict, ResultsDict
import fit_core.parameter as param

# Access default parameters
lcdm_defaults = param.DEFAULTS['lcdm']
pbuf_defaults = param.DEFAULTS['pbuf']

print("LCDM parameters:", lcdm_defaults)
print("PBUF parameters:", pbuf_defaults)
```

## Development Status

✅ **Task 1 Complete**: Project structure and core interfaces  
🔄 **In Progress**: Implementation of core functionality

See `.kiro/specs/pbuf-cosmology-refactor/tasks.md` for detailed implementation plan.

## Requirements

- Python 3.8+
- NumPy >= 1.21.0
- SciPy >= 1.7.0
- Matplotlib >= 3.5.0

Optional dependencies (uncomment in `requirements.txt` as needed):
- pandas, h5py, astropy, pytest, jupyter, sphinx

## License

See LICENSE.txt for details.