#!/bin/bash

# PBUF Cosmology Pipeline Environment Setup Script

echo "🚀 Setting up PBUF Cosmology Pipeline environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core scientific dependencies..."
pip install numpy>=1.21.0 scipy>=1.7.0 matplotlib>=3.5.0

# Test imports
echo "🧪 Testing core module imports..."
python3 -c "
import sys
sys.path.append('pipelines')
from fit_core import ParameterDict, ResultsDict
import fit_core.parameter as param
import fit_core.statistics as stats
print('✅ All core modules imported successfully!')
"

echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the pipeline modules, run:"
echo "  python3 -c \"import sys; sys.path.append('pipelines'); import fit_core; print('Pipeline ready!')\""