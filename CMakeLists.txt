cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

# Giữ đúng tên project cũ
set(PROJECT_NAME face_recogition_tensorRT)
project(${PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# --- OpenCV ---
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# --- CUDA (headers + runtime) ---
find_package(CUDA REQUIRED)
message(STATUS "CUDA: ${CUDA_VERSION}")
include_directories(${CUDA_INCLUDE_DIRS} /usr/local/cuda/include)

# Gợi ý kiến trúc GPU: Orin = sm_87 (Nano = sm_53)
if (NOT DEFINED CUDA_ARCH_BIN)
  set(CUDA_ARCH_BIN 87)
endif()
set(CUDA_NVCC_FLAGS
  ${CUDA_NVCC_FLAGS}
  -O3
  -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN}
)

# --- TensorRT (Jetson layout) ---
include_directories(/usr/include/aarch64-linux-gnu)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include /usr/include/aarch64-linux-gnu
  PATH_SUFFIXES include)
find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib/aarch64-linux-gnu
  PATH_SUFFIXES lib lib64 lib/x64)

set(TENSORRT_LIBRARY
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_INFER_PLUGIN}
)
if (TENSORRT_LIBRARY_PARSER)
  list(APPEND TENSORRT_LIBRARY ${TENSORRT_LIBRARY_PARSER})
endif()

# --- Nguồn mã ---
AUX_SOURCE_DIRECTORY(${CMAKE_SOURCE_DIR}/src DIR_SRCS)
message(STATUS "DIR_SRCS = ${DIR_SRCS}")

add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
)

# --- Linker (thứ tự quan trọng: libs sau target) ---
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}        # kéo libcudart
  ${TENSORRT_LIBRARY}
  Threads::Threads
)

find_package(Threads REQUIRED)

