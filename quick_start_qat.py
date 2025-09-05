#!/usr/bin/env python3
"""
Quick Start Script for QAT Training on Jetson Orin
This script helps you get started with QAT training using your YOLO format data
"""

import os
import sys
import subprocess

def print_header(title):
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n📝 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print("❌ Failed!")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def check_prerequisites():
    """Check if all prerequisites are available (Jetson Orin optimized)"""
    print_header("Checking Prerequisites")
    
    all_good = True
    
    # Check Python packages with Jetson-specific alternatives
    packages_to_check = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('cv2', 'opencv (built-in Jetson)'),
        ('numpy', 'numpy'),
        ('PIL', 'pillow (or alternatives)')
    ]
    
    for import_name, display_name in packages_to_check:
        try:
            if import_name == 'cv2':
                import cv2
                print(f"✅ {display_name} - version {cv2.__version__}")
            elif import_name == 'PIL':
                try:
                    from PIL import Image
                    print(f"✅ {display_name} - PIL available")
                except ImportError:
                    # Try alternative image libraries
                    try:
                        import cv2
                        print(f"⚠️  {display_name} - Using OpenCV for image operations")
                    except ImportError:
                        print(f"❌ {display_name} - Need image processing library")
                        all_good = False
            else:
                module = __import__(import_name)
                if hasattr(module, '__version__'):
                    print(f"✅ {display_name} - version {module.__version__}")
                else:
                    print(f"✅ {display_name}")
        except ImportError:
            if import_name == 'cv2':
                # Try opencv-python as fallback
                try:
                    import cv2
                    print(f"✅ {display_name}")
                except ImportError:
                    print(f"❌ {display_name} - Need to install")
                    all_good = False
            else:
                print(f"❌ {display_name} - Need to install")
                all_good = False
    
    # Check data structure
    print("\n📁 Checking WIDER FACE data structure...")
    
    required_dirs = [
        'qat/data/WIDER_train/images',
        'qat/data/WIDER_val/images', 
        'qat/data/wider_face_split',
        'qat/models',
        'qat/training',
        'qat/utils'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Missing!")
            all_good = False
    
    # Check annotation files
    annotation_files = [
        'qat/data/wider_face_split/wider_face_train_bbx_gt.txt',
        'qat/data/wider_face_split/wider_face_val_bbx_gt.txt'
    ]
    
    for ann_file in annotation_files:
        if os.path.exists(ann_file):
            print(f"✅ {ann_file}")
        else:
            print(f"❌ {ann_file} - Missing!")
            all_good = False
    
    # Check data files
    if os.path.exists('qat/data/WIDER_train/images'):
        train_images = 0
        for root, dirs, files in os.walk('qat/data/WIDER_train/images'):
            train_images += len([f for f in files if f.lower().endswith(('.jpg', '.png'))])
        print(f"📊 Found {train_images} training images")
        
        if train_images == 0:
            print("❌ No training images found!")
            all_good = False
    
    return all_good

def quick_start_training():
    """Run quick start training sequence"""
    print_header("Quick Start QAT Training")
    
    print("This will train all three MTCNN models with QAT:")
    print("1. P-Net (Proposal Network)")
    print("2. R-Net (Refinement Network)") 
    print("3. O-Net (Output Network)")
    print("\nEach model will be trained for a few epochs to verify everything works.")
    
    # Create necessary directories
    os.makedirs('models/qat_checkpoints', exist_ok=True)
    os.makedirs('models/qat_final', exist_ok=True)
    os.makedirs('models/onnx', exist_ok=True)
    
    # Test data loading first
    print("\n🧪 Testing data loading...")
    if not run_command("python test_data_loading.py", "Testing YOLO data loading"):
        print("❌ Data loading test failed. Please fix data issues first.")
        return False
    
    # Train models
    models = ['pnet', 'rnet', 'onet']
    
    for model_type in models:
        print(f"\n🎯 Training {model_type.upper()}...")
        
        cmd = f"python3 train_jetson_qat.py --model {model_type} --data_dir qat/data"
        
        if not run_command(cmd, f"Training {model_type.upper()} with QAT"):
            print(f"❌ {model_type.upper()} training failed!")
            return False
        
        print(f"✅ {model_type.upper()} training completed!")
    
    print("\n🎉 All models trained successfully!")
    return True

def export_to_onnx():
    """Export trained models to ONNX"""
    print_header("Exporting to ONNX")
    
    # Look for trained models
    checkpoint_dir = 'models/qat_checkpoints'
    final_dir = 'models/qat_final'
    
    model_files = []
    for directory in [final_dir, checkpoint_dir]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith('.pth'):
                    model_files.append(os.path.join(directory, file))
    
    if not model_files:
        print("❌ No trained models found. Please train models first.")
        return False
    
    print(f"📁 Found {len(model_files)} trained models")
    
    # Export each model
    for model_file in model_files:
        model_name = os.path.basename(model_file)
        print(f"\n📤 Exporting {model_name}...")
        
        cmd = f"python3 qat/export/export_onnx.py --model_path {model_file}"
        
        if run_command(cmd, f"Exporting {model_name} to ONNX"):
            print(f"✅ {model_name} exported successfully!")
        else:
            print(f"❌ Failed to export {model_name}")
    
    return True

def build_tensorrt_engines():
    """Build TensorRT engines for Jetson"""
    print_header("Building TensorRT Engines")
    
    if not os.path.exists('scripts/build_jetson_engines.sh'):
        print("❌ build_jetson_engines.sh script not found!")
        return False
    
    # Make script executable
    run_command("chmod +x scripts/build_jetson_engines.sh", "Making build script executable")
    
    # Run the build script
    if run_command("bash scripts/build_jetson_engines.sh", "Building TensorRT engines for Jetson"):
        print("✅ TensorRT engines built successfully!")
        return True
    else:
        print("❌ Failed to build TensorRT engines")
        return False

def main():
    print("🚀 QAT MTCNN Quick Start for Jetson Orin")
    print("This script will guide you through the complete QAT training process")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed!")
        print("\nFor Jetson Orin users:")
        print("1. Run: python3 setup_jetson.py")
        print("   This will install only missing packages and use pre-installed libraries")
        print("\nFor other systems:")
        print("pip3 install torch torchvision opencv-python numpy pillow")
        return
    
    print("\n✅ All prerequisites satisfied!")
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Check system and data only")
    print("2. Quick start training (train all models)")
    print("3. Export trained models to ONNX")
    print("4. Build TensorRT engines")
    print("5. Complete pipeline (train → export → build)")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        run_command("python3 train_jetson_qat.py --check_only", "System and data check")
        
    elif choice == '2':
        quick_start_training()
        
    elif choice == '3':
        export_to_onnx()
        
    elif choice == '4':
        build_tensorrt_engines()
        
    elif choice == '5':
        print("\n🔄 Running complete pipeline...")
        
        if quick_start_training():
            if export_to_onnx():
                if build_tensorrt_engines():
                    print("\n🎉 Complete pipeline finished successfully!")
                    print("\nYour INT8 TensorRT engines are ready for deployment!")
                    print("You can now integrate them with your C++ MTCNN code.")
                else:
                    print("\n⚠️  Pipeline partially completed. TensorRT build failed.")
            else:
                print("\n⚠️  Pipeline partially completed. ONNX export failed.")
        else:
            print("\n❌ Pipeline failed at training stage.")
    
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
