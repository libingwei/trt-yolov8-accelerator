#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <memory>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cstring>

#include <trt_utils/trt_runtime.h>
#include <trt_utils/trt_common.h>

#include <opencv2/opencv.hpp>
#include <trt_utils/trt_preprocess.h>
#include <trt_utils/trt_vision.h>
#include <trt_utils/trt_decode.h>

class Logger : public nvinfer1::ILogger{ void log(Severity s, const char* m) noexcept override { if(s<=Severity::kWARNING) std::cout<<m<<"\n"; }};

static std::vector<char> readAll(const std::string& p){ std::ifstream f(p, std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0); std::vector<char> b(n); f.read(b.data(), n); return b; }

// Find first INPUT/OUTPUT name
static std::string firstByMode(const nvinfer1::ICudaEngine& e, nvinfer1::TensorIOMode mode){
	int n=e.getNbIOTensors();
	for(int i=0;i<n;++i){ const char* name=e.getIOTensorName(i); if(e.getTensorIOMode(name)==mode) return std::string(name); }
	return std::string();
}

using TrtVision::Detection;

int main(int argc, char** argv){
	if(argc<2){
		std::cerr<<"Usage: "<<argv[0]<<" <engine.trt> [--image path] [--H 640] [--W 640] [--conf 0.25] [--iou 0.5] [--decode cpu|plugin]\n";
		return 1;
	}
	std::string eng=argv[1]; std::string imagePath; int H=640,W=640; float confTh=0.25f, iouTh=0.5f; int B=1; std::string decode="cpu";
	for(int i=2;i<argc;++i){ std::string t=argv[i];
		if(t=="--image" && i+1<argc) imagePath=argv[++i];
		else if(t=="--H" && i+1<argc) H=std::stoi(argv[++i]);
		else if(t=="--W" && i+1<argc) W=std::stoi(argv[++i]);
		else if(t=="--conf" && i+1<argc) confTh=std::stof(argv[++i]);
		else if(t=="--iou" && i+1<argc) iouTh=std::stof(argv[++i]);
		else if(t=="--decode" && i+1<argc) decode=argv[++i];
	}

	TrtLogger g; TrtRunner runner(g);
	if(!runner.loadEngineFromFile(eng)){ std::cerr<<"Failed to load engine\n"; return 2; }
	if(!runner.prepare(B, H, W)){ std::cerr<<"Failed to prepare runner\n"; return 2; }
	auto outShape = runner.outputDims();
	size_t inSize = runner.inputSize();
	size_t outSize = runner.outputSize();

	std::vector<float> hIn(inSize, 0.5f), hOut(outSize);
	cv::Mat src;
	LetterboxInfo lb{};
	if(!imagePath.empty()){
		src = cv::imread(imagePath);
		if(src.empty()){ std::cerr<<"Failed to read image: "<<imagePath<<"\n"; return 3; }
		cv::Mat f = preprocessLetterboxWithMeanStd(src, W, H, /*toRGB*/true, /*scaleTo01*/true, nullptr, nullptr, cv::Scalar(114,114,114), &lb);
		std::vector<cv::Mat> ch; cv::split(f, ch);
		size_t plane = (size_t)W*H;
		// CHW
		memcpy(hIn.data()+0*plane, ch[0].data, plane*sizeof(float));
		memcpy(hIn.data()+1*plane, ch[1].data, plane*sizeof(float));
		memcpy(hIn.data()+2*plane, ch[2].data, plane*sizeof(float));
	}

	// Run once using default stream; TrtRunner manages device buffers
	runner.run(1, hIn.data(), hOut.data(), /*useDefaultStream*/true);

	// Try to decode YOLOv8 head: support layouts [B,N,C], [B,C,N], [N,C], [C,N]
	if(!imagePath.empty()){
		int nbDims = outShape.nbDims;
		int N=0, C=0; bool layout_NC=true; // true if [..., N, C]
		if(nbDims==3){
			// Prefer [B,N,C] if last dim is small (C) and middle is large (N)
			int d0 = outShape.d[0], d1 = outShape.d[1], d2 = outShape.d[2];
			if(d1> d2){ N=d1; C=d2; layout_NC=true; }
			else { N=d2; C=d1; layout_NC=false; }
		if(N>0 && C>=6){
			int d0 = outShape.d[0], d1 = outShape.d[1];
			if(d0> d1){ N=d0; C=d1; layout_NC=true; }
			else { N=d1; C=d0; layout_NC=false; }
		}
	std::vector<Detection> dets;
	if(N>0 && C>=6){
			const float* p = hOut.data();
			auto get_val = [&](int i, int k){
				// i in [0,N), k in [0,C)
				if(nbDims==3){
					if(layout_NC) return p + i*C + k; // [B,N,C]
			std::vector<Detection> dets;
				} else { // nbDims==2
					if(layout_NC) return p + i*C + k; // [N,C]
				// Flatten to contiguous [N,C]
				std::vector<float> flat(N*C);
				for(int i=0;i<N;++i){ for(int k=0;k<C;++k){ flat[i*C+k] = *get_val(i,k); } }
				TrtDecode::YoloDecodeConfig cfg; cfg.alreadyDecoded = true; cfg.hasObjectness=false; cfg.numClasses=C-5;
				auto ydets = TrtDecode::decode(flat.data(), N, C, cfg, confTh, lb.padX, lb.padY, lb.scale, src.cols, src.rows, W, H);
				dets.reserve(ydets.size()); for(auto& d: ydets) dets.push_back({d.box, d.cls, d.conf});
						x1 = std::max(0.f, nx1); y1=std::max(0.f, ny1);
				std::vector<float> flat(N*C);
				for(int i=0;i<N;++i){ for(int k=0;k<C;++k){ flat[i*C+k] = *get_val(i,k); } }
				TrtDecode::YoloDecodeConfig cfg; cfg.alreadyDecoded = false; cfg.hasObjectness = (C>=85); cfg.numClasses = C - (cfg.hasObjectness?5:4);
				auto ydets = TrtDecode::decode(flat.data(), N, C, cfg, confTh, lb.padX, lb.padY, lb.scale, src.cols, src.rows, W, H);
				dets.reserve(ydets.size()); for(auto& d: ydets) dets.push_back({d.box, d.cls, d.conf});
				float y1 = std::max(0.f, by - bh/2.f);
				float x2 = std::min((float)src.cols-1, bx + bw/2.f);
				float y2 = std::min((float)src.rows-1, by + bh/2.f);
				dets.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), cls, conf});
			}
			auto keep = TrtVision::nms(dets, iouTh, true);
			for(int idx: keep){
				const auto& d = dets[idx];
				cv::rectangle(src, d.box, cv::Scalar(0,255,0), 2);
				std::ostringstream ss; ss<<d.cls<<" "<<std::fixed<<std::setprecision(2)<<d.conf;
				cv::putText(src, ss.str(), d.box.tl()+cv::Point2f(0,-3), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
			}
			cv::imwrite("yolo_out.jpg", src);
			std::cout<<"Saved detections to yolo_out.jpg ("<<keep.size()<<" boxes)\n";
		} else {
			std::cout<<"Output dims not recognized for YOLOv8 head; ran inference only. outSize="<<outSize<<"\n";
		}
	} else {
		std::cout<<"Ran once. in="<<inSize<<" out="<<outSize<<"\n";
	}

	// Runner cleans up in destructor
	return 0;
}
