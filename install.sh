#!/bin/bash
# Installation script for Deep Reasoning LLM

echo "=========================================="
echo "Deep Reasoning LLM - Installation Script"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv || {
    echo "Failed to create venv with python3, trying with system Python..."
    python -m venv venv
}

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Warning: Could not activate virtual environment"
    echo "Please manually activate it with: source venv/bin/activate"
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Prepare sample data:"
echo "   python scripts/prepare_sample_data.py"
echo ""
echo "3. Train tokenizer:"
echo "   python scripts/train_tokenizer.py --data_dir data/raw --output_dir tokenizer"
echo ""
echo "4. Test model configuration:"
echo "   python scripts/test_config.py"
echo ""
echo "5. Start training:"
echo "   python scripts/train.py --config configs/base_config.yaml"
echo ""
echo "For more details, see QUICKSTART.md"
echo "=========================================="
