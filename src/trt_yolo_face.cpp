#include "trt_yolo_face.h"
#include <fstream>
#include <numeric>
#include <algorithm>

using namespace nvinfer1;

static std::vector<unsigned char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open engine: " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
}

void TrtYoloFace::checkCuda(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::exit(1);
    }
}

TrtYoloFace::TrtYoloFace(const std::string& engine_path,
                         float conf_th, float iou_th)
: conf_th_(conf_th), iou_th_(iou_th) {
    // load engine
    auto engineData = readFile(engine_path);
    runtime_ = createInferRuntime(logger_);
    if (!runtime_) throw std::runtime_error("createInferRuntime failed");
    engine_ = runtime_->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");
    context_ = engine_->createExecutionContext();
    if (!context_) throw std::runtime_error("createExecutionContext failed");

    // bindings
    int nb = engine_->getNbBindings();
    for (int i=0;i<nb;i++){
        if (engine_->bindingIsInput(i)) in_bind_ = i;
        else out_bind_ = i;
    }
    in_dims_  = engine_->getBindingDimensions(in_bind_);
    out_dims_ = engine_->getBindingDimensions(out_bind_);

    // input dims: [N,3,H,W] or [3,H,W]
    if (in_dims_.nbDims == 4) { in_w_ = in_dims_.d[3]; in_h_ = in_dims_.d[2]; }
    else if (in_dims_.nbDims == 3) { in_w_ = in_dims_.d[2]; in_h_ = in_dims_.d[1]; }
    else throw std::runtime_error("Unexpected input dims");

    // bytes
    size_t in_count = 1;
    for (int i=0;i<in_dims_.nbDims;i++) in_count *= in_dims_.d[i];
    in_bytes_ = in_count * sizeof(float);

    size_t out_count = 1;
    for (int i=0;i<out_dims_.nbDims;i++) out_count *= out_dims_.d[i];
    out_bytes_ = out_count * sizeof(float);

    // alloc
    CHECK_CUDA(cudaMalloc(&buffers_[in_bind_],  in_bytes_));
    CHECK_CUDA(cudaMalloc(&buffers_[out_bind_], out_bytes_));
}

TrtYoloFace::~TrtYoloFace(){
    if (buffers_[in_bind_])  cudaFree(buffers_[in_bind_]);
    if (buffers_[out_bind_]) cudaFree(buffers_[out_bind_]);
    if (context_) context_->destroy();
    if (engine_)  engine_->destroy();
    if (runtime_) runtime_->destroy();
}

cv::Mat TrtYoloFace::letterbox(const cv::Mat& img, int new_w, int new_h,
                               cv::Scalar color, float& r, int& padw, int& padh)
{
    int w = img.cols, h = img.rows;
    r = std::min((float)new_w / w, (float)new_h / h);
    int nw = (int)std::round(w * r);
    int nh = (int)std::round(h * r);
    padw = (new_w - nw) / 2;
    padh = (new_h - nh) / 2;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(nw, nh));
    cv::Mat out(new_h, new_w, img.type(), color);
    resized.copyTo(out(cv::Rect(padw, padh, nw, nh)));
    return out;
}

static inline float iouOf(const cv::Rect& a, const cv::Rect& b) {
    int interx = std::max(0, std::min(a.x+a.width, b.x+b.width) - std::max(a.x,b.x));
    int intery = std::max(0, std::min(a.y+a.height, b.y+b.height) - std::max(a.y,b.y));
    int inter = interx * intery;
    int uni = a.area() + b.area() - inter;
    return uni>0 ? (float)inter/uni : 0.f;
}
void TrtYoloFace::nms(std::vector<FaceDet>& dets, float iou_th) {
    std::sort(dets.begin(), dets.end(), [](const FaceDet& a, const FaceDet& b){
        return a.conf > b.conf;
    });
    std::vector<char> removed(dets.size(), 0);
    std::vector<FaceDet> keep;
    for (size_t i=0;i<dets.size();++i){
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j=i+1;j<dets.size();++j){
            if (removed[j]) continue;
            if (iouOf(dets[i].box, dets[j].box) > iou_th) removed[j] = 1;
        }
    }
    dets.swap(keep);
}

std::vector<FaceDet> TrtYoloFace::detect(const cv::Mat& bgr)
{
    if (!context_) return {};
    // 1) preprocess (letterbox -> CHW float32 [0,1])
    float r; int padw, padh;
    cv::Mat lb = letterbox(bgr, in_w_, in_h_, cv::Scalar(114,114,114), r, padw, padh);
    cv::Mat rgb; cv::cvtColor(lb, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1/255.0);

    std::vector<float> input(in_w_ * in_h_ * 3);
    // HWC -> CHW
    std::vector<cv::Mat> ch(3);
    for (int i=0;i<3;i++) ch[i] = cv::Mat(in_h_, in_w_, CV_32F, input.data() + i*in_w_*in_h_);
    cv::split(rgb, ch);

    // 2) infer
    CHECK_CUDA(cudaMemcpy(buffers_[in_bind_], input.data(), in_bytes_, cudaMemcpyHostToDevice));
    context_->enqueueV2(buffers_, 0, nullptr);

    std::vector<float> out(out_bytes_ / sizeof(float));
    CHECK_CUDA(cudaMemcpy(out.data(), buffers_[out_bind_], out_bytes_, cudaMemcpyDeviceToHost));

    // 3) postprocess
    // out shape: [1, num, k] hoặc [num,k]
    int num = 0, k = 0;
    if (out_dims_.nbDims == 3) { num = out_dims_.d[1]; k = out_dims_.d[2]; }
    else if (out_dims_.nbDims == 2) { num = out_dims_.d[0]; k = out_dims_.d[1]; }
    else if (out_dims_.nbDims == 4) { num = out_dims_.d[2]; k = out_dims_.d[3]; }
    else { throw std::runtime_error("Unexpected output dims"); }

    std::vector<FaceDet> dets;
    dets.reserve(num);

    for (int i=0;i<num;i++){
        const float* p = &out[i * k];
        float cx, cy, w, h, conf;
        if (YOLO_BOX_IS_XYXY) {
            float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
            conf = p[4];
            if (conf < conf_th_) continue;
            cx = (x1+x2)/2; cy=(y1+y2)/2; w = x2-x1; h=y2-y1;
        } else {
            cx = p[0]; cy = p[1]; w = p[2]; h = p[3];
            conf = p[4];
            if (conf < conf_th_) continue;
        }

        // map về ảnh gốc (bỏ letterbox)
        float x = (cx - w/2.f - padw) / r;
        float y = (cy - h/2.f - padh) / r;
        float ww = w / r;
        float hh = h / r;

        cv::Rect box{ (int)std::round(x), (int)std::round(y),
                      (int)std::round(ww), (int)std::round(hh) };

        FaceDet d; d.box = box; d.conf = conf;

        // landmarks nếu có: p[5..14] = (x1,y1,...,x5,y5)
        if (k >= 15) {
            d.has_landmarks = true;
            for (int j=0;j<5;j++){
                float lx = p[5 + j*2];
                float ly = p[5 + j*2 + 1];
                // bỏ letterbox
                lx = (lx - padw) / r;
                ly = (ly - padh) / r;
                d.lmk[j] = cv::Point2f(lx, ly);
            }
        }
        dets.push_back(std::move(d));
    }

    if (!dets.empty()) nms(dets, iou_th_);
    return dets;
}

