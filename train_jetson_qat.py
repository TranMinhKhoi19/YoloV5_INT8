#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QAT Training Script optimized for Jetson Orin
Trains MTCNN models with WIDER FACE dataset
Python 2/3 compatible version
"""

import os
import sys
import torch
import time
import argparse
from datetime import datetime

# Add the project root to Python path
sys.path.append('qat')

from qat.utils.dataset import MTCNNDataset
from qat.models.mtcnn_qat import PNet, RNet, ONet
from qat.training.train_qat import train_qat_model
from qat.utils.config import QATConfig


def get_jetson_info():
    """Get Jetson system information"""
    info = {}
    
    # Check if running on Jetson
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read().strip().replace('\x00', '')
            if 'jetson' in model.lower():
                info['device'] = model
                info['is_jetson'] = True
            else:
                info['device'] = 'Unknown'
                info['is_jetson'] = False
    except:
        info['device'] = 'Unknown'
        info['is_jetson'] = False
    
    # Check GPU memory
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        info['gpu_name'] = 'No GPU'
        info['gpu_memory'] = 0
    
    # Check system memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemTotal:'):
                    mem_kb = int(line.split()[1])
                    info['system_memory'] = mem_kb / 1024 / 1024  # Convert to GB
                    break
    except:
        info['system_memory'] = 0
    
    return info


def optimize_for_jetson(model_type='pnet'):
    """Get optimized training settings for Jetson Orin"""
    
    jetson_info = get_jetson_info()
    
    # Conservative settings for Jetson Orin
    if jetson_info['is_jetson']:
        settings = {
            'pnet': {
                'batch_size': 16,      # Very conservative for Jetson
                'num_workers': 1,      # Minimal workers to save memory
                'max_epochs': 3,       # Quick training
                'accumulate_grad': 4,  # Larger gradient accumulation
                'pin_memory': False,   # Don't pin memory on Jetson
            },
            'rnet': {
                'batch_size': 8,       # Very small batch
                'num_workers': 1,
                'max_epochs': 5,
                'accumulate_grad': 8,
                'pin_memory': False,
            },
            'onet': {
                'batch_size': 4,       # Minimal batch size
                'num_workers': 1,
                'max_epochs': 8,
                'accumulate_grad': 16,
                'pin_memory': False,
            }
        }
    else:
        # Standard settings for PC
        settings = {
            'pnet': {
                'batch_size': 64,
                'num_workers': 4,
                'max_epochs': 8,
                'accumulate_grad': 1,
                'pin_memory': True,
            },
            'rnet': {
                'batch_size': 32,
                'num_workers': 4,
                'max_epochs': 10,
                'accumulate_grad': 1,
                'pin_memory': True,
            },
            'onet': {
                'batch_size': 16,
                'num_workers': 2,
                'max_epochs': 12,
                'accumulate_grad': 1,
                'pin_memory': True,
            }
        }
    
    return settings.get(model_type, settings['pnet'])


def train_jetson_qat(model_type='pnet', data_dir='qat/data'):
    """Train QAT MTCNN on Jetson Orin with WIDER FACE dataset"""
    
    print("=" * 60)
    print("Starting QAT MTCNN Training for Jetson Orin")
    print("=" * 60)
    
    # System info
    jetson_info = get_jetson_info()
    print("Device: {}".format(jetson_info['device']))
    print("GPU: {}".format(jetson_info['gpu_name']))
    print("GPU Memory: {:.1f} GB".format(jetson_info['gpu_memory']))
    print("System Memory: {:.1f} GB".format(jetson_info['system_memory']))
    print("CUDA Available: {}".format(torch.cuda.is_available()))
    
    # Get optimized settings
    settings = optimize_for_jetson(model_type)
    
    print("\nTraining {} with optimized Jetson settings:".format(model_type.upper()))
    for key, value in settings.items():
        print("  - {}: {}".format(key, value))
    
    # Check data availability
    train_images_dir = os.path.join(data_dir, 'WIDER_train', 'images')
    annotation_file = os.path.join(data_dir, 'wider_face_split', 'wider_face_train_bbx_gt.txt')
    
    if not os.path.exists(train_images_dir):
        print("[ERROR] Missing: {}".format(train_images_dir))
        return False
    
    if not os.path.exists(annotation_file):
        print("[ERROR] Missing: {}".format(annotation_file))
        return False
    
    # Count training images
    train_images = 0
    for root, dirs, files in os.walk(train_images_dir):
        train_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print("[OK] Found {} training images".format(train_images))
    print("[OK] Found annotation file: {}".format(os.path.basename(annotation_file)))
    
    # Create dataset
    print("\nCreating dataset...")
    try:
        dataset = MTCNNDataset(
            data_dir=data_dir,
            split='train',
            model_type=model_type
        )
        print("[OK] Dataset created: {} samples".format(len(dataset)))
    except Exception as e:
        print("[ERROR] Dataset creation failed: {}".format(str(e)))
        return False
    
    # Create model
    print("\nCreating {} model...".format(model_type.upper()))
    try:
        if model_type == 'pnet':
            model = PNet(use_qat=True)
        elif model_type == 'rnet':
            model = RNet(use_qat=True)
        elif model_type == 'onet':
            model = ONet(use_qat=True)
        else:
            raise ValueError("Unknown model type: {}".format(model_type))
        
        # Prepare QAT
        model.prepare_qat()
        print("[OK] Model created and QAT prepared")
    except Exception as e:
        print("[ERROR] Model creation failed: {}".format(str(e)))
        return False
    
    # Training
    print("\nStarting training...")
    start_time = time.time()
    
    try:
        # Setup training config
        config = QATConfig()
        config.batch_size = settings['batch_size']
        config.num_workers = settings['num_workers']
        config.max_epochs = settings['max_epochs']
        config.accumulate_grad = settings['accumulate_grad']
        config.pin_memory = settings['pin_memory']
        
        # Train model
        trained_model = train_qat_model(
            model=model,
            dataset=dataset,
            config=config,
            model_type=model_type
        )
        
        training_time = time.time() - start_time
        print("[OK] Training completed in {:.1f} minutes".format(training_time/60))
        
        # Save model
        os.makedirs('models/qat_final', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = "models/qat_final/{}_qat_{}.pth".format(model_type, timestamp)
        torch.save(trained_model.state_dict(), model_path)
        print("[OK] Model saved: {}".format(model_path))
        
        return True
        
    except Exception as e:
        print("[ERROR] Training failed: {}".format(str(e)))
        return False


def check_system_only():
    """Only check system and data, don't train"""
    print("=" * 60)
    print("System and Data Check for Jetson Orin")
    print("=" * 60)
    
    # System info
    jetson_info = get_jetson_info()
    print("Device: {}".format(jetson_info['device']))
    print("Is Jetson: {}".format(jetson_info['is_jetson']))
    print("GPU: {}".format(jetson_info['gpu_name']))
    print("GPU Memory: {:.1f} GB".format(jetson_info['gpu_memory']))
    print("System Memory: {:.1f} GB".format(jetson_info['system_memory']))
    print("CUDA Available: {}".format(torch.cuda.is_available()))
    
    # Check PyTorch version
    print("PyTorch Version: {}".format(torch.__version__))
    
    # Check data structure
    data_dir = 'qat/data'
    required_paths = [
        'qat/data/WIDER_train/images',
        'qat/data/wider_face_split/wider_face_train_bbx_gt.txt'
    ]
    
    print("\nData Structure Check:")
    for path in required_paths:
        if os.path.exists(path):
            print("[OK] {}".format(path))
        else:
            print("[MISSING] {}".format(path))
    
    # Count images
    train_images_dir = os.path.join(data_dir, 'WIDER_train', 'images')
    if os.path.exists(train_images_dir):
        train_images = 0
        for root, dirs, files in os.walk(train_images_dir):
            train_images += len([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        print("Training Images: {}".format(train_images))
    
    print("\n[OK] System and data check completed")


def main():
    parser = argparse.ArgumentParser(description='QAT Training for Jetson Orin')
    parser.add_argument('--model', type=str, default='pnet', 
                       choices=['pnet', 'rnet', 'onet'],
                       help='Model type to train')
    parser.add_argument('--data_dir', type=str, default='qat/data',
                       help='Data directory path')
    parser.add_argument('--check_only', action='store_true',
                       help='Only check system and data, don\'t train')
    
    args = parser.parse_args()
    
    if args.check_only:
        check_system_only()
    else:
        success = train_jetson_qat(args.model, args.data_dir)
        if success:
            print("\n[SUCCESS] Training completed successfully!")
        else:
            print("\n[FAILED] Training failed. Check error messages above.")


if __name__ == '__main__':
    main()
