@echo off
echo Setting up QAT MTCNN Training Environment...

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv qat_env

:: Activate virtual environment
echo Activating virtual environment...
call qat_env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch (CPU version for initial setup, adjust for your hardware)
echo Installing PyTorch...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

:: Install other requirements
echo Installing additional requirements...
pip install onnx onnxruntime opencv-python numpy Pillow tensorboard tqdm matplotlib scikit-learn lmdb h5py scipy pandas

:: Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import onnx; print(f'ONNX version: {onnx.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo.
echo QAT Environment setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   qat_env\Scripts\activate.bat
echo.
echo To start training, first prepare your WIDER FACE dataset in the data directory,
echo then run:
echo   python qat\training\train_qat.py --model all
echo.
pause
