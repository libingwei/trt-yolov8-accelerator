#include "NvInfer.h"
#include <vector>
#include <string>
#include <iostream>
#include <glob.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <fstream>

#define CHECK(status) \
    do { auto ret = (status); if (ret != 0) { \
        std::cerr << "CUDA failure: " << cudaGetErrorString(ret) << std::endl; abort(); } } while(0)

static std::vector<std::string> glob_files(const std::string& pattern){
    glob_t g; std::vector<std::string> out; 
    if(::glob(pattern.c_str(), 0, nullptr, &g)==0){
        for(size_t i=0;i<g.gl_pathc;++i) out.emplace_back(g.gl_pathv[i]);
    }
    globfree(&g); return out;
}

class YoloInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    YoloInt8Calibrator(int bs, int W, int H, const std::string& dir, const std::string& cache)
    :batch(bs),W(W),H(H),dir(dir),cache(cache){
        imgs = glob_files(dir+"/*.jpg");
        auto pngs = glob_files(dir+"/*.png"); imgs.insert(imgs.end(), pngs.begin(), pngs.end());
        if(imgs.empty()) { std::cerr<<"No calibration images in "<<dir<<"\n"; }
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
            cv::Mat r; cv::resize(img, r, cv::Size(W,H)); r.convertTo(r, CV_32FC3, 1.0/255.0);
            std::vector<cv::Mat> ch; cv::split(r, ch);
            float* dst = host.data()+ i*inputCount;
            size_t plane=W*H; memcpy(dst, ch[2].data, plane*sizeof(float)); // RGB
            memcpy(dst+plane, ch[1].data, plane*sizeof(float));
            memcpy(dst+2*plane, ch[0].data, plane*sizeof(float));
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
};

// Factory (simple headerless approach for demo). Real project可拆到头文件。
std::unique_ptr<nvinfer1::IInt8Calibrator> createYoloCalibrator(int bs,int W,int H,const std::string& dir,const std::string& cache){
    return std::unique_ptr<nvinfer1::IInt8Calibrator>(new YoloInt8Calibrator(bs,W,H,dir,cache));
}
