#pragma once
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <array>
#include <string>
#include <vector>

// ======= cấu hình YOLOv5-face =======
// Giả định engine đã "decode" sẵn đầu ra (toạ độ tuyệt đối):
// mỗi dòng: [cx, cy, w, h, conf, (lmk_x1,lmk_y1,...,lmk_x5,lmk_y5)?]
// Nếu engine của bạn đưa ra [x1,y1,x2,y2,conf,...] hãy set YOLO_BOX_IS_XYXY = 1
// Nếu engine KHÔNG decode, bạn sẽ cần thêm bước grid-decode (không khuyến nghị với export Ultralytics).
static const bool YOLO_BOX_IS_XYXY = false;

struct FaceDet {
    cv::Rect box;
    float conf{0.f};
    std::array<cv::Point2f,5> lmk{};
    bool has_landmarks{false};
};

class TrtYoloFace {
public:
    TrtYoloFace(const std::string& engine_path,
                float conf_th=0.30f, float iou_th=0.45f);
    ~TrtYoloFace();

    std::vector<FaceDet> detect(const cv::Mat& bgr);

    inline int input_w()  const { return in_w_; }
    inline int input_h()  const { return in_h_; }
    inline bool valid()   const { return context_!=nullptr; }

private:
    // TensorRT
    class TRTLogger : public nvinfer1::ILogger {
        void log(Severity s, const char* msg) noexcept override {
            if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
        }
    } logger_;
    nvinfer1::IRuntime*       runtime_{nullptr};
    nvinfer1::ICudaEngine*    engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};

    int in_bind_{-1}, out_bind_{-1};
    nvinfer1::Dims in_dims_{}, out_dims_{};

    void* buffers_[2]{nullptr,nullptr};   // [input, output]
    size_t in_bytes_{0}, out_bytes_{0};

    // preprocess / postprocess
    int in_w_{640}, in_h_{640};
    float conf_th_, iou_th_;

    cv::Mat letterbox(const cv::Mat& img, int new_w, int new_h,
                      cv::Scalar color, float& r, int& padw, int& padh);
    void nms(std::vector<FaceDet>& dets, float iou_th);

    // cuda helpers
    void checkCuda(cudaError_t e, const char* file, int line);
};

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) this->checkCuda((x), __FILE__, __LINE__)
#endif

