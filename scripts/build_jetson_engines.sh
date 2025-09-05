#!/bin/bash

# TensorRT Engine Builder for Jetson Orin - Optimized for AGX Orin
# This script builds INT8 engines from QAT ONNX models with Jetson-specific optimizations

set -e  # Exit on any error

echo "🚀 Building TensorRT INT8 Engines for Jetson AGX Orin..."

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "⚠️  Warning: This script is optimized for Jetson devices"
    echo "   For x86 systems, use the generic build script instead"
fi

# Check if trtexec is available
if ! command -v trtexec &> /dev/null; then
    echo "❌ trtexec not found. Please ensure TensorRT is installed."
    echo "   For Jetson, TensorRT should be included with JetPack"
    exit 1
fi

# Create output directory for engines
ENGINE_DIR="../models/mtcnn_int8"
mkdir -p $ENGINE_DIR

echo "📁 Engine output directory: $ENGINE_DIR"

# Jetson AGX Orin specific settings
WORKSPACE_SIZE=2048  # Reduced for Jetson memory constraints
MAX_BATCH_SIZE=4     # Conservative batch size for real-time inference
TIMING_CACHE="timing_cache.trt"

# Check available GPU memory
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "🖥️  Available GPU Memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -lt 8192 ]; then
    echo "⚠️  GPU memory < 8GB detected. Using conservative settings..."
    WORKSPACE_SIZE=1024
    MAX_BATCH_SIZE=2
fi

# Common trtexec flags for Jetson Orin
COMMON_FLAGS=(
    --int8
    --workspace=$WORKSPACE_SIZE
    --maxBatch=$MAX_BATCH_SIZE
    --timingCacheFile=$TIMING_CACHE
    --saveEngine
    --verbose
    --profilingVerbosity=detailed
    --dumpLayerInfo
    --dumpProfile
    --separateProfileRun
    --avgRuns=10
    # Jetson specific optimizations
    --useCudaGraph
    --useSpinWait
    --threads
    # Memory optimizations
    --memPoolSize=workspace:1024
    --plugins
)

# Function to build engine with error handling
build_engine() {
    local model_name=$1
    local onnx_file=$2
    local engine_file=$3
    local input_shapes=$4
    
    echo ""
    echo "🔨 Building $model_name engine..."
    echo "   Input: $onnx_file"
    echo "   Output: $engine_file"
    echo "   Input shapes: $input_shapes"
    
    # Add input shapes if provided
    local shape_flags=""
    if [ ! -z "$input_shapes" ]; then
        shape_flags="--shapes=$input_shapes"
    fi
    
    # Build command
    local cmd="trtexec --onnx=$onnx_file --saveEngine=$engine_file $shape_flags ${COMMON_FLAGS[*]}"
    
    echo "   Command: $cmd"
    echo ""
    
    # Execute with timeout (max 10 minutes per model)
    if timeout 600 $cmd; then
        echo "✅ $model_name engine built successfully!"
        
        # Verify engine file
        if [ -f "$engine_file" ]; then
            local size=$(du -h "$engine_file" | cut -f1)
            echo "   Engine size: $size"
        else
            echo "❌ Engine file not found after build"
            return 1
        fi
    else
        echo "❌ Failed to build $model_name engine (timeout or error)"
        return 1
    fi
}

# Build P-Net engine
if [ -f "pnet_qat_qdq.onnx" ]; then
    build_engine "P-Net" "pnet_qat_qdq.onnx" "$ENGINE_DIR/pnet_int8.plan" "input:1x3x48x48"
else
    echo "⚠️  P-Net ONNX file not found: pnet_qat_qdq.onnx"
fi

# Build R-Net engine  
if [ -f "rnet_qat_qdq.onnx" ]; then
    build_engine "R-Net" "rnet_qat_qdq.onnx" "$ENGINE_DIR/rnet_int8.plan" "input:1x3x24x24"
else
    echo "⚠️  R-Net ONNX file not found: rnet_qat_qdq.onnx"
fi

# Build O-Net engine
if [ -f "onet_qat_qdq.onnx" ]; then
    build_engine "O-Net" "onet_qat_qdq.onnx" "$ENGINE_DIR/onet_int8.plan" "input:1x3x48x48"
else
    echo "⚠️  O-Net ONNX file not found: onet_qat_qdq.onnx"
fi

echo ""
echo "🎉 TensorRT Engine Building Complete!"
echo ""
echo "📊 Engine Summary:"
echo "=================="

for engine in $ENGINE_DIR/*.plan; do
    if [ -f "$engine" ]; then
        local name=$(basename "$engine")
        local size=$(du -h "$engine" | cut -f1)
        echo "  $name: $size"
    fi
done

echo ""
echo "📝 Performance Optimization Tips for Jetson:"
echo "============================================"
echo "1. Set performance mode: sudo nvpmodel -m 0"
echo "2. Max clocks: sudo jetson_clocks"
echo "3. Increase swap if needed: sudo systemctl enable nvzramconfig"
echo "4. Monitor temperature: watch nvidia-smi"
echo ""
echo "🔗 Next Steps:"
echo "=============="
echo "1. Copy engines to C++ project:"
echo "   cp $ENGINE_DIR/*.plan ../mtCNNModels/"
echo ""
echo "2. Update C++ code to use new engines (if needed)"
echo ""
echo "3. Compile and test:"
echo "   cd .. && mkdir -p build && cd build"
echo "   cmake .. && make"
echo "   ./face_recogition_tensorRT"
echo ""
echo "✨ Your QAT MTCNN models are ready for deployment on Jetson Orin!"
