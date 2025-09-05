# QAT Training cho MTCNN trên Jetson Orin

## Tổng quan
Bạn đã có sẵn WIDER FACE dataset hoàn chỉnh và QAT implementation. Hệ thống sẽ sử dụng WIDER FACE dataset để train MTCNN với Quantization-Aware Training.

## Cấu trúc Dataset hiện tại
```
qat/data/
├── WIDER_train/
│   └── images/             # ✅ Training images (theo categories)
│       ├── 0--Parade/
│       ├── 1--Handshaking/
│       └── ...
├── WIDER_val/
│   └── images/             # ✅ Validation images
├── WIDER_test/
│   └── images/             # ✅ Test images  
└── wider_face_split/
    ├── wider_face_train_bbx_gt.txt  # ✅ Training annotations
    └── wider_face_val_bbx_gt.txt    # ✅ Validation annotations
```

## Quick Start - Chỉ 3 bước

### Bước 1: Kiểm tra WIDER FACE dataset
```bash
python test_wider_face.py
```

### Bước 2: Train tất cả models
```bash
python quick_start_qat.py  
# Chọn option 5: Complete pipeline (train → export → build)
```

### Bước 3: Sử dụng với C++ code
Sau khi train xong, các file INT8 engine sẽ được tạo trong `mtCNNModels/`:
- `det1_relu_int8.engine` (P-Net)
- `det2_relu_int8.engine` (R-Net)  
- `det3_relu_int8.engine` (O-Net)

Copy chúng vào thư mục gốc và compile C++ code như bình thường.

## Training riêng lẻ (nếu cần)

### Train từng model:
```bash
# Train P-Net (5 epochs, ~10-15 phút)
python train_jetson_qat.py --model pnet --data_dir qat/data

# Train R-Net (8 epochs, ~15-20 phút)  
python train_jetson_qat.py --model rnet --data_dir qat/data

# Train O-Net (10 epochs, ~20-25 phút)
python train_jetson_qat.py --model onet --data_dir qat/data
```

### Export to ONNX:
```bash
python qat/export/export_onnx.py --model_path models/qat_final/pnet_qat_*.pth
```

### Build TensorRT:
```bash
bash scripts/build_jetson_engines.sh
```

## Tối ưu cho Jetson Orin

Hệ thống đã được tối ưu sẵn:
- **Batch size nhỏ**: 32→16→8 (P-Net→R-Net→O-Net)
- **Memory conservative**: Gradient accumulation để tiết kiệm RAM
- **Ít workers**: 2→2→1 để tránh overload
- **Epochs ngắn**: 5→8→10 để train nhanh

## Thời gian ước tính

**Trên Jetson Orin:**
- P-Net: ~10-15 phút
- R-Net: ~15-20 phút  
- O-Net: ~20-25 phút
- **Tổng**: ~45-60 phút

**Trên PC (nếu train trước):**
- P-Net: ~5-8 phút
- R-Net: ~8-12 phút
- O-Net: ~10-15 phút  
- **Tổng**: ~25-35 phút

## Hiệu suất dự kiến

Sau khi áp dụng QAT + TensorRT INT8:
- **Tốc độ**: 2-3x nhanh hơn (30 FPS → 60-90 FPS)
- **Memory**: Giảm ~50% (4GB → 2GB)  
- **Accuracy**: Giữ nguyên >95%

## Troubleshooting

### Out of Memory:
```bash
# Giảm batch size hơn nữa
python train_jetson_qat.py --model pnet  # Sẽ tự động dùng batch size nhỏ
```

### Data loading error:
```bash
# Test WIDER FACE dataset trước
python test_wider_face.py
```

### Missing packages:
```bash
pip install torch torchvision opencv-python numpy pillow
```

## Files quan trọng

- `quick_start_qat.py` - Main script, chọn option 5
- `train_jetson_qat.py` - Training script tối ưu cho Jetson
- `test_wider_face.py` - Test WIDER FACE dataset format
- `qat/` - Toàn bộ QAT implementation  
- `scripts/build_jetson_engines.sh` - Build TensorRT engines

## Kết quả

Sau khi hoàn thành:
```
mtCNNModels/
├── det1_relu_int8.engine  # ✅ P-Net INT8
├── det2_relu_int8.engine  # ✅ R-Net INT8  
└── det3_relu_int8.engine  # ✅ O-Net INT8
```

Chỉ cần thay thế các engine files trong C++ code và compile lại!
