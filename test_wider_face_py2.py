#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WIDER FACE Dataset Testing Script for QAT Training (Python 2.7 Compatible)
Tests dataset structure, annotation files, and image loading.
"""

import os
import sys

def test_wider_face_structure(data_dir="qat/data"):
    """Test if WIDER FACE dataset structure is correct."""
    print("Testing WIDER FACE dataset structure...")
    print("Base directory: {}".format(data_dir))
    
    # Required directories and files
    required_dirs = [
        "WIDER_train/images",
        "wider_face_split"
    ]
    
    required_files = [
        "wider_face_split/wider_face_train_bbx_gt.txt"
    ]
    
    # Check directories
    print("\nChecking directories:")
    for dir_name in required_dirs:
        dir_path = os.path.join(data_dir, dir_name)
        if os.path.exists(dir_path):
            print("[OK] {}".format(dir_name))
            
            if 'images' in dir_name:
                # Count images in subdirectories
                image_count = 0
                for root, dirs, files in os.walk(dir_path):
                    image_count += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                print("   - {} images".format(image_count))
                
                # List some subdirectories
                try:
                    subdirs = os.listdir(dir_path)[:5]  # First 5 subdirs
                    print("   - Sample subdirs: {}".format(subdirs))
                except OSError:
                    print("   - Could not list subdirectories")
        else:
            print("[MISSING] {} - NOT FOUND".format(dir_name))
    
    # Check annotation files
    print("\nChecking annotation files:")
    for file_name in required_files:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / 1024.0 / 1024.0  # MB
            print("[OK] {} ({:.1f} MB)".format(file_name, file_size))
        else:
            print("[MISSING] {} - NOT FOUND".format(file_name))
    
    return True

def test_annotation_parsing(data_dir="qat/data"):
    """Test parsing of WIDER FACE annotation file."""
    print("\nTesting annotation parsing...")
    
    annotation_file = os.path.join(data_dir, "wider_face_split", "wider_face_train_bbx_gt.txt")
    
    if not os.path.exists(annotation_file):
        print("[ERROR] Annotation file not found: {}".format(annotation_file))
        return False
    
    try:
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        print("[OK] Loaded {} lines from annotation file".format(len(lines)))
        
        # Parse first few entries
        annotations = {}
        i = 0
        entry_count = 0
        
        while i < len(lines) and entry_count < 3:
            line = lines[i].strip()
            if line.endswith('.jpg'):
                img_name = line
                i += 1
                if i < len(lines):
                    num_faces = int(lines[i].strip())
                    i += 1
                    
                    faces = []
                    for j in range(num_faces):
                        if i + j < len(lines):
                            bbox_line = lines[i + j].strip()
                            if bbox_line:
                                parts = bbox_line.split()
                                if len(parts) >= 4:
                                    x, y, w, h = map(int, parts[:4])
                                    faces.append([x, y, w, h])
                    
                    annotations[img_name] = faces
                    print("   - {}: {} faces".format(img_name, len(faces)))
                    i += num_faces
                    entry_count += 1
            else:
                i += 1
        
        print("[OK] Successfully parsed {} image annotations".format(len(annotations)))
        return True
        
    except Exception as e:
        print("[ERROR] Failed to parse annotations: {}".format(str(e)))
        return False

def test_dataset_loading(data_dir="qat/data"):
    """Test loading dataset with custom dataset class."""
    print("\nTesting dataset loading...")
    
    try:
        # Try to import the dataset class
        sys.path.append('qat/utils')
        from dataset import MTCNNDataset
        
        # Test P-Net dataset
        print("Testing P-Net dataset...")
        pnet_dataset = MTCNNDataset(
            data_dir=data_dir,
            stage='pnet',
            input_size=12
        )
        
        print("[OK] P-Net dataset created with {} samples".format(len(pnet_dataset)))
        
        # Test loading a few samples
        if len(pnet_dataset) > 0:
            sample = pnet_dataset[0]
            print("   - Sample shape: {}".format(sample[0].shape))
            print("   - Label shape: {}".format(sample[1].shape))
        
        return True
        
    except ImportError as e:
        print("[WARNING] Cannot import dataset class: {}".format(str(e)))
        print("Make sure you're in the project root directory")
        return False
    except Exception as e:
        print("[ERROR] Dataset loading failed: {}".format(str(e)))
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("WIDER FACE Dataset Verification for QAT Training")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("qat"):
        print("[ERROR] qat/ directory not found!")
        print("Please run this script from the project root directory")
        return False
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_wider_face_structure():
        tests_passed += 1
    
    if test_annotation_parsing():
        tests_passed += 1
    
    if test_dataset_loading():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    print("TEST RESULTS: {}/{} tests passed".format(tests_passed, total_tests))
    
    if tests_passed == total_tests:
        print("[SUCCESS] WIDER FACE dataset is ready for QAT training!")
        print("\nNext steps:")
        print("1. Run: python3 quick_start_qat.py")
        print("2. Select option 5 for complete training pipeline")
        print("\nNote: Use python3 for training scripts as they require Python 3.6+")
    else:
        print("[WARNING] Some tests failed. Please check the dataset structure.")
        print("\nExpected structure:")
        print("qat/data/")
        print("├── WIDER_train/images/")
        print("└── wider_face_split/wider_face_train_bbx_gt.txt")
    
    print("=" * 60)
    return tests_passed == total_tests

if __name__ == "__main__":
    main()
