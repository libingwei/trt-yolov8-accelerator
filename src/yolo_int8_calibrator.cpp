#include "NvInfer.h"
#include <vector>
#include <string>
#include <iostream>
#include <glob.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <cstdlib>

#include <trt_utils/trt_common.h>
#include <trt_utils/trt_preprocess.h>

// Use CHECK from trt_utils/trt_common.h

// Use shared file collection from trt_utils

class YoloInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    YoloInt8Calibrator(int bs, int W, int H, const std::string& dir, const std::string& cache)
    :batch(bs),W(W),H(H),dir(dir),cache(cache){
    // Collect images with common extensions (case-insensitive); allow recursive via env
    bool recursive = false; if (const char* e = std::getenv("CALIB_RECURSIVE")) { std::string v=e; if (v=="1"||v=="true") recursive=true; }
    imgs = TrtHelpers::collectImages(dir, {"jpg","JPG","jpeg","JPEG","png","PNG"}, recursive);
        if(imgs.empty()) { std::cerr<<"No calibration images in "<<dir<<" (supported: .jpg/.jpeg/.png)\n"; }

        // Read preprocess options from env (reuse shared convention)
        if (const char* e = std::getenv("IMAGENET_CENTER_CROP")) {
            std::string v = e; if (v=="1" || v=="true") optCenterCrop = true;
        }
    // YOLO mean/std normalization (default disabled; enable via env)
    yoloUseMeanStd = false; // 默认关闭，符合常见 YOLO 预处理
        yoloMean = cv::Scalar(0.0, 0.0, 0.0);
        yoloStd  = cv::Scalar(1.0, 1.0, 1.0);
        auto parseScalar3 = [](const std::string& s, cv::Scalar& out)->bool{
            std::stringstream ss(s);
            std::string item; std::vector<double> v; v.reserve(3);
            while (std::getline(ss, item, ',')) {
                if(item.empty()) continue; v.push_back(std::stod(item));
            }
            if (v.size()!=3) return false;
            // clamp std to avoid division by zero later
            out = cv::Scalar(v[0], v[1], v[2]);
            return true;
        };
        if (const char* m = std::getenv("YOLO_MEAN")) {
            cv::Scalar tmp; if (parseScalar3(m, tmp)) { yoloMean = tmp; yoloUseMeanStd = true; }
        }
        if (const char* s = std::getenv("YOLO_STD")) {
            cv::Scalar tmp; if (parseScalar3(s, tmp)) {
                // 防止 0 标准差
                auto eps = 1e-12; tmp[0] = std::max(tmp[0], eps); tmp[1] = std::max(tmp[1], eps); tmp[2] = std::max(tmp[2], eps);
                yoloStd = tmp; yoloUseMeanStd = true;
            }
        }

        inputCount = 3*W*H; host.resize(batch*inputCount);
        CHECK(cudaMalloc(&device, batch*inputCount*sizeof(float)));
    }
    ~YoloInt8Calibrator(){ CHECK(cudaFree(device)); }
    int getBatchSize() const noexcept override { return batch; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if(cur>=imgs.size()) return false;
        int n=std::min<int>(batch, imgs.size()-cur);
        for(int i=0;i<n;++i){
            cv::Mat img = cv::imread(imgs[cur+i]); if(img.empty()) continue;
            // 使用通用可配置预处理：centerCrop(可选)、BGR->RGB、缩放[0,1]、(img-mean)/std
            const cv::Scalar* meanPtr = yoloUseMeanStd ? &yoloMean : nullptr;
            const cv::Scalar* stdPtr  = yoloUseMeanStd ? &yoloStd  : nullptr;
            cv::Mat f = preprocessImageWithMeanStd(img, W, H, optCenterCrop, /*toRGB*/true, /*scaleTo01*/true, meanPtr, stdPtr);
            float* dst = host.data()+ i*inputCount;
            hwcToChw(f, dst);
        }
        CHECK(cudaMemcpy(device, host.data(), n*inputCount*sizeof(float), cudaMemcpyHostToDevice));
        bindings[0]=device; cur+=n; return true;
    }
    const void* readCalibrationCache(size_t& length) noexcept override {
        cacheBuf.clear();
        std::ifstream f(cache, std::ios::binary); if(f){ cacheBuf.assign((std::istreambuf_iterator<char>(f)), {}); }
        length = cacheBuf.size(); return length? cacheBuf.data(): nullptr;
    }
    void writeCalibrationCache(const void* c, size_t length) noexcept override {
        std::ofstream f(cache, std::ios::binary); f.write((const char*)c, length);
    }
private:
    int batch, W, H; std::string dir, cache; size_t inputCount; size_t cur{0};
    std::vector<std::string> imgs; std::vector<float> host; void* device{nullptr}; std::vector<char> cacheBuf;
    bool optCenterCrop{false};
};

// Factory (simple headerless approach for demo). Real project可拆到头文件。
std::unique_ptr<nvinfer1::IInt8Calibrator> createYoloCalibrator(int bs,int W,int H,const std::string& dir,const std::string& cache){
    return std::unique_ptr<nvinfer1::IInt8Calibrator>(new YoloInt8Calibrator(bs,W,H,dir,cache));
}
