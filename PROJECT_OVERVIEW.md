# Face Recognition with QAT MTCNN - Project Overview

This project implements a complete face recognition system with Quantization-Aware Training (QAT) for MTCNN models, optimized for deployment on NVIDIA Jetson devices with TensorRT INT8 inference.

## 🚀 Key Features

- **Real-time Face Recognition**: MTCNN face detection + FaceNet feature extraction
- **QAT Optimization**: INT8 quantization with minimal accuracy loss
- **TensorRT Integration**: Optimized inference on NVIDIA hardware
- **Smooth Tracking**: Advanced face tracking with jitter reduction
- **Edge Deployment**: Optimized for Jetson AGX Orin and similar devices

## 📁 Project Structure

```
face_recog/
├── src/                    # C++ implementation (existing)
│   ├── main.cpp           # Main application
│   ├── faceNet.cpp        # FaceNet inference with tracking
│   ├── mtcnn.cpp          # MTCNN face detection
│   └── ...                # Other C++ source files
├── qat/                   # QAT training pipeline (NEW)
│   ├── models/            # PyTorch MTCNN models with QAT
│   ├── training/          # Training scripts and utilities
│   ├── export/            # ONNX export with Q/DQ operators
│   ├── data/              # Dataset preparation (WIDER FACE)
│   └── utils/             # Common utilities
├── models/                # Model storage
│   ├── mtcnn_fp32/        # Original FP32 models
│   ├── mtcnn_qat/         # QAT trained models and ONNX
│   ├── mtcnn_int8/        # TensorRT INT8 engines
│   └── facenetModels/     # FaceNet models (existing)
├── scripts/               # Build and deployment scripts
├── imgs/                  # Face database for recognition
└── build/                 # Compiled binaries
```

## 🔧 Development Roadmap

### Phase 1: QAT Infrastructure Setup ✅
- [x] Create organized directory structure
- [x] Setup configuration files and documentation
- [x] Create environment setup scripts
- [x] Define requirements and dependencies

### Phase 2: PyTorch MTCNN Implementation (In Progress)
- [ ] Implement base MTCNN models (P-Net, R-Net, O-Net)
- [ ] Add QAT support with proper quantization configuration
- [ ] Create custom layers and building blocks
- [ ] Implement multi-task loss functions

### Phase 3: Dataset Preparation
- [ ] WIDER FACE dataset preparation pipeline
- [ ] Data loaders for MTCNN training
- [ ] Data augmentation and preprocessing
- [ ] Annotation parsing utilities

### Phase 4: QAT Training Pipeline
- [ ] P-Net QAT training implementation
- [ ] R-Net QAT training with P-Net proposals
- [ ] O-Net QAT training with R-Net proposals
- [ ] Training orchestration and hyperparameter tuning

### Phase 5: Model Export and Optimization
- [ ] ONNX export with Q/DQ operators
- [ ] TensorRT INT8 engine building
- [ ] Model validation and accuracy verification
- [ ] Performance benchmarking

### Phase 6: C++ Integration
- [ ] Update C++ MTCNN implementation for INT8 models
- [ ] Integrate new TensorRT engines
- [ ] Performance optimization and testing
- [ ] Deploy and validate on Jetson hardware

## 🎯 Performance Targets

| Metric | Current (FP32) | Target (INT8) | Improvement |
|--------|----------------|---------------|-------------|
| **Speed** | ~30 FPS | ~60-90 FPS | 2-3x faster |
| **Memory** | ~800 MB | ~400 MB | 50% reduction |
| **Accuracy** | Baseline | >95% retained | Minimal loss |
| **Power** | Baseline | 40-60% reduction | Significant savings |

## 🛠 Getting Started

### Prerequisites
- NVIDIA Jetson AGX Orin (or compatible GPU)
- JetPack 5.1+ with TensorRT 8.5+
- Python 3.8+ with PyTorch 2.0+
- CMake 3.16+ for C++ compilation

### Quick Setup
```bash
# 1. Setup Python environment
cd scripts
./setup_environment.sh  # Linux/Mac
# or
setup_environment.bat   # Windows

# 2. Activate environment
source qat_env/bin/activate

# 3. Prepare dataset (when ready)
cd ../qat/data
python prepare_wider_face.py --data_root /path/to/WIDER_FACE

# 4. Train QAT models (when implemented)
cd ../training
python train_pnet_qat.py
python train_rnet_qat.py  
python train_onet_qat.py

# 5. Export to TensorRT (when ready)
cd ../export
python export_onnx_qdq.py --model all
```

## 📊 Current Status

### Completed ✅
- Project structure and organization
- Documentation and configuration setup
- Environment setup scripts
- Development roadmap planning

### In Progress 🚧
- PyTorch MTCNN model implementation
- QAT integration and configuration
- Dataset preparation pipeline

### Planned 📋
- Training pipeline implementation
- Model export and TensorRT integration
- C++ integration and deployment
- Performance optimization and validation

## 🤝 Contributing

This is an internal development project. For questions or suggestions, please refer to the documentation in each subdirectory.

## 📄 License

This project builds upon existing MTCNN and FaceNet implementations. Please respect the original licenses and citations.

## 🙏 Acknowledgments

- Original MTCNN implementation by [PKUZHOU](https://github.com/PKUZHOU/MTCNN_FaceDetection_TensorRT)
- FaceNet architecture by Google Research
- QAT methodology from PyTorch documentation
- WIDER FACE dataset by Chinese University of Hong Kong
