#!/bin/bash

# Jetson Orin Environment Setup for QAT MTCNN
# Optimized for JetPack 5.1+ with TensorRT 8.5+

echo "🚀 Setting up QAT MTCNN Environment on Jetson Orin..."

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "❌ This script is designed for Jetson devices"
    echo "   Use setup_environment.sh for PC installation"
    exit 1
fi

# Display Jetson info
echo "📊 Jetson Device Information:"
cat /etc/nv_tegra_release
echo ""

# Check JetPack version
if command -v jetson_release &> /dev/null; then
    jetson_release
else
    echo "⚠️  jetson_release not found. Install with: sudo pip3 install jetson-stats"
fi

echo ""

# Check available resources
echo "💾 System Resources:"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo "Storage: $(df -h / | tail -1 | awk '{print $4}') available"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "   Install with: sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

echo "🐍 Python version: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv qat_env

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source qat_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch for Jetson (special version)
echo "🔥 Installing PyTorch for Jetson..."
echo "   Note: Using Jetson-optimized PyTorch wheel"

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    # Install PyTorch wheel for Jetson (JetPack 5.1+)
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
    
    # Alternative: Install from NVIDIA's pre-built wheels if needed
    # pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
else
    echo "❌ Unexpected architecture: $ARCH"
    echo "   This script is designed for aarch64 (ARM64) Jetson devices"
    exit 1
fi

# Install ONNX and ONNX Runtime for Jetson
echo "🧠 Installing ONNX and ONNX Runtime..."
pip install onnx

# Install ONNX Runtime GPU for Jetson (if available)
pip install onnxruntime-gpu || pip install onnxruntime

# Install other dependencies (Jetson-optimized versions where possible)
echo "📚 Installing additional dependencies..."

# OpenCV is usually pre-installed on Jetson, but ensure Python bindings
pip install opencv-python || echo "⚠️  Using system OpenCV (recommended for Jetson)"

# Scientific computing libraries
pip install numpy scipy pandas matplotlib seaborn scikit-learn

# Deep learning utilities
pip install tensorboard tqdm Pillow h5py lmdb

# Development tools
pip install jupyter notebook

# Jetson-specific monitoring tools
pip install jetson-stats

echo ""
echo "🔍 Verifying installation..."

# Verify PyTorch installation
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Verify ONNX
python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"

# Verify ONNX Runtime
python -c "
import onnxruntime as ort
print(f'ONNX Runtime version: {ort.__version__}')
providers = ort.get_available_providers()
print(f'Available providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('✅ GPU acceleration available')
else:
    print('⚠️  CPU-only ONNX Runtime')
"

# Verify OpenCV
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "🎯 Jetson-specific optimizations..."

# Set performance mode (requires sudo)
echo "🚀 To maximize performance, run these commands:"
echo "   sudo nvpmodel -m 0          # Max performance mode"
echo "   sudo jetson_clocks           # Max clock speeds"
echo "   sudo systemctl disable nvzramconfig  # Disable zram if low on memory"
echo ""

# Create Jetson-specific configuration
cat > qat_jetson_config.py << EOF
# Jetson Orin QAT Configuration
JETSON_CONFIG = {
    'device': 'cuda',
    'mixed_precision': True,
    'batch_size_multiplier': 0.5,  # Reduce batch size for Jetson
    'num_workers': 2,              # Conservative for ARM CPU
    'pin_memory': True,
    'persistent_workers': True,
    
    # TensorRT specific
    'tensorrt_precision': 'int8',
    'workspace_size': 2048,  # MB, conservative for Jetson
    'max_batch_size': 4,
    
    # Memory management
    'gradient_checkpointing': True,
    'empty_cache_freq': 10,  # Clear cache every 10 batches
}
EOF

echo "📝 Created Jetson-specific configuration: qat_jetson_config.py"

echo ""
echo "✅ QAT Environment setup completed successfully on Jetson Orin!"
echo ""
echo "🔗 Next Steps:"
echo "=============="
echo "1. Activate environment: source qat_env/bin/activate"
echo ""
echo "2a. If training on Jetson (slower):"
echo "    python qat/training/train_qat.py --model all --batch_size 16"
echo ""
echo "2b. If using pre-trained models from PC:"
echo "    # Copy ONNX models from PC, then:"
echo "    cd models/mtcnn_qat"
echo "    ../../scripts/build_jetson_engines.sh"
echo ""
echo "3. Integrate with C++ project:"
echo "   cp models/mtcnn_int8/*.plan mtCNNModels/"
echo "   cd build && make && ./face_recogition_tensorRT"
echo ""
echo "📊 Monitor performance with:"
echo "   jtop                    # Real-time system monitoring"
echo "   nvidia-smi             # GPU monitoring"
echo "   tegrastats             # Jetson-specific stats"
echo ""
echo "🎉 Ready for high-performance face recognition on Jetson Orin!"
