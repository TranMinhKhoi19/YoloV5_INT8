# QAT MTCNN Implementation Guide

## Overview

This document provides a comprehensive guide for implementing Quantization-Aware Training (QAT) for MTCNN in your face recognition project. The implementation follows the methodology described in `doc.txt` and integrates with your existing TensorRT C++ inference pipeline.

## Quick Start

### 1. Environment Setup

```bash
# Windows
cd scripts
setup_environment.bat

# Linux/Mac  
cd scripts
chmod +x setup_environment.sh
./setup_environment.sh

# Activate environment
source qat_env/bin/activate  # Linux/Mac
# or
qat_env\Scripts\activate.bat  # Windows
```

### 2. Data Preparation

```bash
# Download WIDER FACE dataset to data/wider_face/
# Structure should be:
# data/
#   wider_face/
#     images/
#       train/
#       val/
#       test/
#     wider_face_train_bbx_gt.txt
#     wider_face_val_bbx_gt.txt
```

### 3. Train QAT Models

```bash
# Train all models (P-Net, R-Net, O-Net)
python qat/training/train_qat.py --model all --num_epochs 10

# Train individual models
python qat/training/train_qat.py --model pnet --num_epochs 8
python qat/training/train_qat.py --model rnet --num_epochs 10  
python qat/training/train_qat.py --model onet --num_epochs 12
```

### 4. Export ONNX Models

```bash
# Export all trained models to ONNX with Q/DQ operators
python qat/export/export_onnx.py --checkpoint_dir checkpoints --output_dir models/mtcnn_qat --generate_trt_script
```

### 5. Build TensorRT Engines

```bash
# Run the generated script to build INT8 engines
cd models/mtcnn_qat
./build_tensorrt_engines.sh
```

## Integration with Existing C++ Code

### Option 1: Replace Existing Engines (Recommended)

1. **Backup existing models:**
```bash
cp mtCNNModels/det1_relu.engine mtCNNModels/det1_relu_fp32_backup.engine
cp mtCNNModels/det2_relu.engine mtCNNModels/det2_relu_fp32_backup.engine  
cp mtCNNModels/det3_relu.engine mtCNNModels/det3_relu_fp32_backup.engine
```

2. **Replace with QAT INT8 engines:**
```bash
cp models/mtcnn_int8/pnet_int8.plan mtCNNModels/det1_relu.engine
cp models/mtcnn_int8/rnet_int8.plan mtCNNModels/det2_relu.engine
cp models/mtcnn_int8/onet_int8.plan mtCNNModels/det3_relu.engine
```

3. **Update thresholds in C++ code if needed:**
```cpp
// In src/mtcnn.cpp, you may need to adjust thresholds after QAT
nms_threshold[0] = 0.6;  // P-Net - may need adjustment
nms_threshold[1] = 0.7;  // R-Net - may need adjustment  
nms_threshold[2] = 0.7;  // O-Net - may need adjustment
```

### Option 2: Modify C++ Code for New Engines

Update the engine paths in your C++ code to point to the new INT8 engines:

```cpp
// In relevant files (pnet_rt.cpp, rnet_rt.cpp, onet_rt.cpp)
// Change engine paths from:
string engineFile = "../mtCNNModels/det1_relu.engine";
// To:
string engineFile = "../models/mtcnn_int8/pnet_int8.plan";
```

## Project Structure After QAT Implementation

```
face_recog/
├── qat/                          # QAT implementation (NEW)
│   ├── models/mtcnn_qat.py      # PyTorch MTCNN models with QAT
│   ├── training/train_qat.py    # Training script
│   ├── export/export_onnx.py    # ONNX export utilities
│   ├── utils/                   # Utilities
│   │   ├── dataset.py          # Dataset loaders
│   │   ├── losses.py           # Loss functions
│   │   └── config.py           # Configuration
│   └── requirements.txt         # Python dependencies
├── models/                      # Model storage (UPDATED)
│   ├── mtcnn_qat/              # QAT ONNX models with Q/DQ
│   └── mtcnn_int8/             # TensorRT INT8 engines
├── src/                         # C++ implementation (EXISTING)
│   ├── main.cpp                # Main application
│   ├── mtcnn.cpp               # May need threshold adjustments
│   └── ...
├── scripts/                     # Setup scripts (NEW)
│   ├── setup_environment.bat   # Windows setup
│   └── setup_environment.sh    # Linux/Mac setup
├── checkpoints/                 # Training checkpoints (NEW)
├── logs/                        # Training logs (NEW)
└── ...                         # Existing files
```

## Performance Expectations

| Metric | Original FP32 | QAT INT8 | Improvement |
|--------|---------------|----------|-------------|
| **Inference Speed** | Baseline | 2-3x faster | Significant |
| **Memory Usage** | Baseline | ~50% reduction | Major |
| **Accuracy** | 100% | >95% retained | Minimal loss |
| **Power Consumption** | Baseline | 40-60% reduction | Substantial |

## Troubleshooting

### Common Issues

1. **Import errors when running training:**
   - Ensure the QAT environment is activated
   - Check that all dependencies are installed

2. **ONNX export fails:**
   - Verify PyTorch and ONNX versions are compatible
   - Check that the model was properly trained with QAT

3. **TensorRT engine build fails:**
   - Ensure TensorRT is installed and in PATH
   - Check ONNX model compatibility with your TensorRT version
   - Verify CUDA toolkit is properly installed

4. **Accuracy degradation after QAT:**
   - Increase training epochs for QAT
   - Adjust learning rate (typically lower for QAT)
   - Fine-tune threshold values in C++ code

5. **C++ compilation errors:**
   - Ensure TensorRT libraries are properly linked
   - Check that engine file paths are correct

### Threshold Tuning

After QAT, you may need to adjust detection thresholds:

```cpp
// In src/mtcnn.cpp
// Original thresholds
nms_threshold[0] = 0.7;  // P-Net
nms_threshold[1] = 0.7;  // R-Net  
nms_threshold[2] = 0.7;  // O-Net

// Potentially adjusted thresholds after QAT
nms_threshold[0] = 0.65; // P-Net - slightly lower
nms_threshold[1] = 0.7;  // R-Net - may stay same
nms_threshold[2] = 0.75; // O-Net - slightly higher
```

Test different threshold combinations and measure precision/recall on your validation set.

## Advanced Usage

### Custom Training Configuration

Modify `qat/utils/config.py` to adjust training parameters:

```python
# Example customizations
MODELS = {
    'pnet': {
        'batch_size': 128,        # Increase if you have more GPU memory
        'num_epochs': 15,         # More epochs for better convergence
        'learning_rate': 5e-5,    # Lower LR for fine-tuning
    }
}

QAT_CONFIG = {
    'freeze_observer_epoch': 5,   # Later freeze for better calibration
    'freeze_bn_epoch': 8,         # Later BN freeze
}
```

### Multi-GPU Training

```bash
# Use multiple GPUs if available
python -m torch.distributed.launch --nproc_per_node=2 qat/training/train_qat.py --model all
```

### Validation and Testing

```bash
# Validate QAT models before deployment
python qat/utils/validate_qat.py --model_dir models/mtcnn_qat --dataset_dir data/wider_face
```

## Integration Timeline

1. **Week 1**: Environment setup and WIDER FACE dataset preparation
2. **Week 2**: P-Net QAT training and validation  
3. **Week 3**: R-Net QAT training with P-Net proposals
4. **Week 4**: O-Net QAT training and integration testing
5. **Week 5**: TensorRT engine building and C++ integration
6. **Week 6**: Performance testing and threshold optimization

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in the `logs/` directory
3. Validate ONNX models with the provided utilities
4. Test individual components step by step

This implementation provides a production-ready QAT pipeline for your MTCNN face recognition system with minimal changes to your existing C++ inference code.
