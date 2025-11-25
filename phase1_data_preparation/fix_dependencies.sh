#!/bin/bash
# Quick fix for dependency issues

echo "Installing compatible dependencies..."
echo

# Activate venv if it exists
if [ -d "../venv" ]; then
    echo "Using project venv..."
    source ../venv/bin/activate
else
    echo "Installing in current environment..."
fi

# Uninstall problematic packages
echo "Cleaning up old packages..."
pip uninstall -y sentence-transformers transformers huggingface-hub numpy 2>/dev/null

# Install fresh compatible versions
echo "Installing compatible versions..."
pip install --no-cache-dir -r requirements.txt

echo
echo "✓ Dependencies installed!"
echo
echo "Test with:"
echo "  cd scripts"
echo "  python demo_queries.py"

