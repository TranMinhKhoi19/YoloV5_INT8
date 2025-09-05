#!/usr/bin/env python3
"""
Jetson Orin Setup Script for QAT Training
Installs only necessary packages that aren't pre-installed
"""

import os
import subprocess
import sys

def check_jetson():
    """Check if running on Jetson"""
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().replace('\x00', '')
            return 'jetson' in model.lower()
    except:
        return False

def install_minimal_packages():
    """Install only essential packages for Jetson"""
    
    print("🚀 Setting up QAT Training on Jetson Orin")
    print("=" * 50)
    
    if not check_jetson():
        print("⚠️  Warning: Not running on Jetson device")
    
    # Check what's already available
    print("\n📋 Checking pre-installed packages...")
    
    packages_status = {}
    
    # Check PyTorch (usually pre-installed on Jetson)
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Already installed")
        packages_status['torch'] = True
    except ImportError:
        print("❌ PyTorch - Need to install")
        packages_status['torch'] = False
    
    # Check OpenCV (usually pre-installed on Jetson)
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__} - Already installed")
        packages_status['cv2'] = True
    except ImportError:
        print("❌ OpenCV - Need to install")
        packages_status['cv2'] = False
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} - Already installed")
        packages_status['numpy'] = True
    except ImportError:
        print("❌ NumPy - Need to install")
        packages_status['numpy'] = False
    
    # Only install what's missing
    missing_packages = []
    
    if not packages_status.get('numpy', False):
        missing_packages.append('numpy')
    
    # For PIL, we can work around it with OpenCV
    try:
        from PIL import Image
        print("✅ PIL - Already installed")
    except ImportError:
        if packages_status.get('cv2', False):
            print("⚠️  PIL not found, but OpenCV available - will use OpenCV for image operations")
        else:
            print("❌ Need image processing library")
            missing_packages.append('pillow')
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {missing_packages}")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages, 
                         check=True)
            print("✅ Installation completed!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    else:
        print("\n✅ All required packages are already available!")
    
    return True

def setup_environment():
    """Setup environment for training"""
    print("\n🔧 Setting up training environment...")
    
    # Create necessary directories
    dirs_to_create = [
        'qat/checkpoints',
        'qat/logs',
        'qat/exports'
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    print("\n🎯 Environment setup complete!")
    print("\nNext steps:")
    print("1. python test_wider_face.py  # Verify dataset")
    print("2. python3 train_jetson_qat.py --model pnet  # Start training")

def main():
    print("Jetson Orin QAT Setup")
    print("=" * 30)
    
    if install_minimal_packages():
        setup_environment()
        print("\n🚀 Ready for QAT training on Jetson Orin!")
    else:
        print("\n❌ Setup failed. Please check error messages above.")

if __name__ == "__main__":
    main()
