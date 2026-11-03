#!/usr/bin/env bash
# Sets up virtual environment, installs dependencies, and registers Jupyter kernel.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Setting up virtual environment in $PROJECT_ROOT"

# Create .venv if not present
if [ ! -d ".venv" ]; then
    echo "Creating .venv..."
    python3 -m venv .venv
else
    echo "Existing .venv found."
fi

# Activate and install packages
source .venv/bin/activate
echo "Using Python: $(which python)"

python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
python -m pip install pytest --quiet

# Register kernel for Jupyter / VS Code
python -m ipykernel install --user --name=aml-dl-venv --display-name="Python (.venv - AML/DL)"

echo "Environment setup complete."
