#!/bin/bash

echo "Setting up QAT MTCNN Training Environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and add it to your PATH"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv qat_env

# Activate virtual environment
echo "Activating virtual environment..."
source qat_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch (adjust for your hardware - this is CPU version)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (uncomment and adjust based on your CUDA version):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing additional requirements..."
pip install onnx onnxruntime opencv-python numpy Pillow tensorboard tqdm matplotlib scikit-learn lmdb h5py scipy pandas

# For GPU inference with ONNX Runtime (uncomment if needed):
# pip install onnxruntime-gpu

# Verify installation
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "QAT Environment setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source qat_env/bin/activate"
echo ""
echo "To start training, first prepare your WIDER FACE dataset in the data directory,"
echo "then run:"
echo "  python qat/training/train_qat.py --model all"
echo ""
