#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

#include "trt_yolo_face.h"
#include "align.h"
#include "faceNet.h"

// ==== Alias Facenet repo ====
using FaceNet = FaceNetClassifier;

// ==== Tiện ích file/camera ====
static bool file_exists(const std::string& p){
    struct stat st; return stat(p.c_str(), &st) == 0;
}

static cv::VideoCapture open_capture_usb(int index=0, int w=640, int h=480, int fps=30){
    cv::VideoCapture cap(index);
    if (cap.isOpened()){
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        cap.set(cv::CAP_PROP_FPS,          fps);
    }
    return cap;
}

static cv::VideoCapture open_capture_csi(int sensor_id=0){
    return cv::VideoCapture(
      "nvarguscamerasrc sensor_id=" + std::to_string(sensor_id) +
      " ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! "
      "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
      "video/x-raw, format=BGR ! appsink",
      cv::CAP_GSTREAMER);
}

// ==== Enroll DB ====
struct Enroll { std::string name; std::vector<float> emb; };
static std::vector<Enroll> g_db;

static float cosine(const std::vector<float>& a, const std::vector<float>& b){
    float s=0, na=0, nb=0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i=0;i<n;++i){ s+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i]; }
    return s / (std::sqrt(na)*std::sqrt(nb) + 1e-9f);
}

static void load_db(const std::string& path="data/embeddings.csv"){
    g_db.clear();
    std::ifstream f(path);
    if(!f) { std::cerr<<"[warn] no embeddings.csv -> YOLO-only\n"; return; }
    std::string line;
    while (std::getline(f, line)){
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string name; std::getline(ss, name, ',');
        std::vector<float> emb; emb.reserve(512);
        for (int i=0;i<512;i++){ std::string v; if(!std::getline(ss,v,',')) break; emb.push_back(std::stof(v)); }
        if (!name.empty() && !emb.empty()) g_db.push_back({name, emb});
    }
    std::cerr<<"[info] loaded "<<g_db.size()<<" identities\n";
}

// CHW float [0..1]
static void to_chw_float01(const cv::Mat& bgr160, std::vector<float>& out) {
    CV_Assert(bgr160.type()==CV_8UC3 && bgr160.cols==160 && bgr160.rows==160);
    out.resize(3*160*160);
    size_t plane = 160*160;
    for (int y=0; y<160; ++y) {
        const uint8_t* p = bgr160.ptr<uint8_t>(y);
        for (int x=0; x<160; ++x) {
            uint8_t b = p[3*x+0], g = p[3*x+1], r = p[3*x+2];
            out[0*plane + y*160 + x] = b/255.0f;
            out[1*plane + y*160 + x] = g/255.0f;
            out[2*plane + y*160 + x] = r/255.0f;
        }
    }
}

// ==== MAIN ====
int main(int argc, char** argv)
{
    // --- Đường dẫn engine ---
    std::string yolo_engine    = "models/yolov5nu_trained_export3.engine";
    std::string facenet_engine = "facenetModels/facenet.engine";
    std::string source         = "usb";  // default

    if (argc>=2) yolo_engine    = argv[1];
    if (argc>=3) facenet_engine = argv[2];
    if (argc>=4) source         = argv[3];

    // --- Camera ---
    cv::VideoCapture cap;
    if (source=="csi") {
        cap = open_capture_csi(0);
        if(!cap.isOpened()){
            std::cerr<<"[warn] CSI không mở được, fallback USB\n";
            cap = open_capture_usb(0);
        }
    } else {
        cap = open_capture_usb(0);
    }
    if(!cap.isOpened()){
        throw std::runtime_error("Không mở được camera");
    }

    // --- YOLO ---
    TrtYoloFace yolo(yolo_engine, 0.30f, 0.45f);
    if(!yolo.valid()){ throw std::runtime_error("YOLO engine không hợp lệ: " + yolo_engine); }

    // --- Facenet ---
    bool yolo_only = !file_exists(facenet_engine);
    FaceNet* faceNet = nullptr;
    if (!yolo_only){
        Logger logger;
        nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
        faceNet = new FaceNet(logger, dtype, "", facenet_engine,
                              1, false, 1.0f, 3, 160, 160);
        load_db();
    } else {
        std::cerr << "[info] YOLO-only mode\n";
    }

    const float MATCH_TH = 0.55f;
    cv::Mat frame;
    while (true) {
        if (!cap.read(frame)) break;
        auto dets = yolo.detect(frame);

        for (auto& d : dets) {
            cv::rectangle(frame, d.box, cv::Scalar(0,255,0), 2);

            if (faceNet){
                cv::Mat face = (d.has_landmarks)
                    ? align_by_5pts(frame, d.lmk, 160, 160)
                    : crop_with_margin(frame, d.box, 0.25f, 160, 160);

                std::vector<float> in_chw; to_chw_float01(face, in_chw);
                std::vector<float> emb(512, 0.0f);
                faceNet->doInference(in_chw.data(), emb.data());

                std::string name = "Unknown";
                float best=-1.f;
                for (auto& e : g_db){
                    float c = cosine(emb,e.emb);
                    if (c>best){ best=c; name=e.name; }
                }
                if (best<MATCH_TH || g_db.empty()) name="Unknown";

                char buf[128]; snprintf(buf,sizeof(buf),"%s (%.2f)", name.c_str(), d.conf);
                int baseline=0; cv::Size t=cv::getTextSize(buf,0,0.6,2,&baseline);
                cv::rectangle(frame,{d.box.x,d.box.y-t.height-6,t.width+6,t.height+6},cv::Scalar(0,255,0),cv::FILLED);
                cv::putText(frame,buf,{d.box.x+3,d.box.y-6},0,0.6,cv::Scalar(0,0,0),2);
            } else {
                char buf[64]; snprintf(buf,sizeof(buf),"face %.2f",d.conf);
                int baseline=0; cv::Size t=cv::getTextSize(buf,0,0.6,2,&baseline);
                cv::rectangle(frame,{d.box.x,d.box.y-t.height-6,t.width+6,t.height+6},cv::Scalar(0,255,0),cv::FILLED);
                cv::putText(frame,buf,{d.box.x+3,d.box.y-6},0,0.6,cv::Scalar(0,0,0),2);
            }
        }

        cv::imshow("YOLO Face Detection", frame);
        if (cv::waitKey(1)==27) break; // ESC
    }
    delete faceNet;
    return 0;
}

