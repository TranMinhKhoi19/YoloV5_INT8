#!/usr/bin/env python3
"""
Test script to verify YOLO format data loading for QAT training
"""

import os
import sys
sys.path.append('qat')

from qat.utils.dataset import MTCNNDataset
import torch
from torch.utils.data import DataLoader


def test_data_loading():
    """Test loading YOLO format data"""
    
    # Check data structure
    data_dir = 'data'
    print("=== Data Structure Check ===")
    
    if os.path.exists(os.path.join(data_dir, 'images', 'train')):
        print("✓ Found images/train directory")
        train_images = len(os.listdir(os.path.join(data_dir, 'images', 'train')))
        print(f"  - {train_images} training images")
    else:
        print("✗ Missing images/train directory")
        return False
        
    if os.path.exists(os.path.join(data_dir, 'labels2')):
        print("✓ Found labels2 directory")
        labels = len([f for f in os.listdir(os.path.join(data_dir, 'labels2')) if f.endswith('.txt')])
        print(f"  - {labels} label files")
    else:
        print("✗ Missing labels2 directory")
        return False
    
    # Test dataset loading
    print("\n=== Dataset Loading Test ===")
    try:
        # Test P-Net dataset
        pnet_dataset = MTCNNDataset(
            data_dir=data_dir,
            split='train',
            model_type='pnet',
            augment=False
        )
        
        print(f"✓ P-Net dataset loaded successfully")
        print(f"  - {len(pnet_dataset)} training samples")
        
        # Test a few samples
        if len(pnet_dataset) > 0:
            sample = pnet_dataset[0]
            print(f"  - Sample keys: {list(sample.keys())}")
            
            # Create DataLoader
            dataloader = DataLoader(pnet_dataset, batch_size=4, shuffle=True, num_workers=0)
            batch = next(iter(dataloader))
            print(f"  - Batch shapes: {batch['image'].shape if 'image' in batch else 'No image key'}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_annotation_format():
    """Test reading annotation files"""
    print("\n=== Annotation Format Test ===")
    
    labels_dir = os.path.join('data', 'labels2')
    sample_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')][:3]
    
    for filename in sample_files:
        filepath = os.path.join(labels_dir, filename)
        print(f"\nFile: {filename}")
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines[:2]):  # First 2 lines only
            line = line.strip()
            if line and 'Human face' in line:
                parts = line.split()
                if len(parts) >= 5:
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    w, h = x2 - x1, y2 - y1
                    print(f"  Face {i+1}: x1={x1:.1f}, y1={y1:.1f}, w={w:.1f}, h={h:.1f}")


if __name__ == "__main__":
    print("Testing YOLO Format Data Loading for QAT Training")
    print("=" * 50)
    
    # Test annotation format
    test_annotation_format()
    
    # Test data loading
    success = test_data_loading()
    
    if success:
        print("\n🎉 Data loading test PASSED!")
        print("\nYou can now proceed with QAT training:")
        print("python qat/training/train_qat.py --data_dir data --model_type pnet --epochs 5")
    else:
        print("\n❌ Data loading test FAILED!")
        print("Please check the data structure and format.")
