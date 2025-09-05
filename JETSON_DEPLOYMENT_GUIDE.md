# QAT MTCNN Deployment Guide for Jetson Orin

## 🎯 Deployment Strategy for Jetson Orin

### **Option 1: Train on PC, Deploy on Jetson (Recommended)**

**Training Phase (PC/Workstation):**
- Train QAT models trên máy có GPU mạnh (RTX 3080/4080+)
- Export ONNX models với Q/DQ operators
- Transfer sang Jetson để build TensorRT engines

**Deployment Phase (Jetson Orin):**
- Build TensorRT INT8 engines từ ONNX
- Integrate với existing C++ inference code
- Run real-time face recognition

### **Option 2: Full Pipeline on Jetson (Slower but Self-contained)**

**All-in-One trên Jetson Orin:**
- Train QAT models trực tiếp trên Jetson (chậm hơn)
- Build engines và deploy

---

## 🚀 **Recommended Workflow**

### **PHASE 1: Training on PC (2-4 hours)**

```bash
# 1. Setup trên PC
git clone <your-repo>
cd "face recog"
scripts/setup_environment.sh

# 2. Download WIDER FACE dataset
# Tải về data/wider_face/

# 3. Train QAT models
source qat_env/bin/activate
python qat/training/train_qat.py --model all --num_epochs 10

# 4. Export ONNX models
python qat/export/export_onnx.py --checkpoint_dir checkpoints --output_dir models/mtcnn_qat
```

### **PHASE 2: Transfer to Jetson Orin (10 minutes)**

```bash
# Copy trained models to Jetson
scp -r models/mtcnn_qat/ jetson@<jetson-ip>:/path/to/face_recog/models/
scp -r checkpoints/ jetson@<jetson-ip>:/path/to/face_recog/
```

### **PHASE 3: Deploy on Jetson Orin (30 minutes)**

```bash
# 1. Setup environment trên Jetson
ssh jetson@<jetson-ip>
cd /path/to/face_recog
scripts/setup_environment.sh

# 2. Build TensorRT INT8 engines
cd models/mtcnn_qat
./build_tensorrt_engines.sh

# 3. Integrate với C++ code
cp ../models/mtcnn_int8/*.plan ../mtCNNModels/

# 4. Compile và test
cd ../..
mkdir -p build && cd build
cmake .. && make
./face_recogition_tensorRT
```

---

## 🔧 **Jetson Orin Specific Optimizations**

### **1. Hardware Configuration**

```bash
# Maximize Jetson performance
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Max clock speeds

# Check available memory
free -h
nvidia-smi        # GPU memory
```

### **2. TensorRT Settings for Jetson**

```bash
# Jetson-optimized TensorRT build (using our custom script)
cd models/mtcnn_qat
../../scripts/build_jetson_engines.sh  # Optimized for Jetson Orin
```

### **3. Memory Management for Jetson**

```bash
# Check memory usage
free -h
nvidia-smi

# Increase swap if needed (for training)
sudo systemctl enable nvzramconfig
sudo systemctl start nvzramconfig

# Monitor during training/inference
watch -n 1 'free -h && nvidia-smi'
```

---

## 💡 **What You Should Run First on Jetson**

### **Scenario A: You Have a Powerful PC (Recommended)**

**Step 1 - Train on PC (2-4 hours):**
```bash
# On your PC
cd "face recog"
scripts/setup_environment.sh
source qat_env/bin/activate

# Download WIDER FACE dataset
python qat/training/train_qat.py --model all --num_epochs 10
python qat/export/export_onnx.py --generate_trt_script
```

**Step 2 - Transfer to Jetson (5 minutes):**
```bash
# Copy only the trained models (not dataset)
scp -r models/mtcnn_qat/ jetson@<jetson-ip>:/path/to/face_recog/models/
```

**Step 3 - Deploy on Jetson (15 minutes):**
```bash
# On Jetson
ssh jetson@<jetson-ip>
cd /path/to/face_recog
scripts/setup_jetson_environment.sh
source qat_env/bin/activate
scripts/build_jetson_engines.sh
```

### **Scenario B: Only Have Jetson (Slower but Possible)**

**Step 1 - Full Setup on Jetson:**
```bash
# On Jetson Orin
git clone <your-repo>
cd "face recog"
scripts/setup_jetson_environment.sh
source qat_env/bin/activate
```

**Step 2 - Train with Jetson-optimized settings:**
```bash
# Reduced batch size and epochs for Jetson
python qat/training/train_qat.py --model all --batch_size 8 --num_epochs 6
```

---

## ⚡ **Performance Optimization Tips**

### **1. Jetson Performance Modes**
```bash
# Check available power modes
sudo nvpmodel -q

# Set maximum performance (Mode 0)
sudo nvpmodel -m 0
sudo jetson_clocks

# For power-efficient mode (Mode 2)
sudo nvpmodel -m 2
```

### **2. TensorRT Optimization**
```bash
# Use Jetson-optimized TensorRT build script
cd models/mtcnn_qat
../../scripts/build_jetson_engines.sh  # Instead of generic script
```

### **3. C++ Integration Notes**

**Threshold Adjustments for INT8:**
```cpp
// In src/mtcnn.cpp - you may need to fine-tune these after QAT
nms_threshold[0] = 0.6;   // P-Net (was 0.7)
nms_threshold[1] = 0.7;   // R-Net 
nms_threshold[2] = 0.75;  // O-Net (was 0.7)
```

**Engine Path Updates:**
```cpp
// Option 1: Replace existing engines
cp models/mtcnn_int8/pnet_int8.plan mtCNNModels/det1_relu.engine

// Option 2: Update paths in C++ code
// Change in pnet_rt.cpp, rnet_rt.cpp, onet_rt.cpp:
string engineFile = "../models/mtcnn_int8/pnet_int8.plan";
```

---

## 🔍 **Compatibility Check**

### **What Works on Jetson Orin:**
- ✅ **QAT Training**: Hoàn toàn có thể (chậm hơn PC)
- ✅ **ONNX Export**: Hoàn toàn hỗ trợ
- ✅ **TensorRT INT8**: Optimal performance target
- ✅ **C++ Integration**: Seamless với existing code
- ✅ **Real-time Inference**: 60-90 FPS expected

### **Requirements:**
- ✅ **JetPack 5.1+** (includes TensorRT 8.5+)
- ✅ **CUDA 11.4+** (included in JetPack)
- ✅ **32GB storage** minimum (for models + dataset)
- ✅ **Sufficient cooling** (for sustained performance)

---

## 🎯 **Recommended Timeline**

| Task | PC Training | Jetson Only |
|------|-------------|-------------|
| **Environment Setup** | 10 min | 15 min |
| **Data Preparation** | 30 min | 30 min |
| **QAT Training** | 2-4 hours | 6-12 hours |
| **ONNX Export** | 5 min | 5 min |
| **TensorRT Build** | 10 min | 15 min |
| **C++ Integration** | 5 min | 5 min |
| **Testing** | 10 min | 10 min |
| **Total** | **~4 hours** | **~8-14 hours** |

---

## 🚨 **Important Notes**

1. **Memory**: QAT training cần ~6-8GB RAM. Jetson Orin có đủ.
2. **Storage**: WIDER FACE dataset ~3GB, models ~500MB
3. **Cooling**: Ensure adequate cooling cho sustained training
4. **Power**: Use Mode 0 cho training, có thể dùng Mode 2 cho inference
5. **Monitoring**: Use `jtop` để monitor temperature và performance

---

## 🎉 **Expected Results on Jetson Orin**

| Metric | FP32 Baseline | QAT INT8 Target |
|--------|---------------|-----------------|
| **Inference Speed** | ~30 FPS | **60-90 FPS** |
| **Memory Usage** | ~800 MB | **~400 MB** |
| **Power Draw** | ~25W | **~15W** |
| **Accuracy** | 100% | **>95%** |
| **Model Size** | ~50MB | **~15MB** |

**Recommendation**: Bắt đầu với **Scenario A** (train trên PC) nếu có PC mạnh, sau đó transfer sang Jetson. Điều này sẽ nhanh hơn và hiệu quả hơn nhiều!
