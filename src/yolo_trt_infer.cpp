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

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include <opencv2/opencv.hpp>

class Logger : public nvinfer1::ILogger{ void log(Severity s, const char* m) noexcept override { if(s<=Severity::kWARNING) std::cout<<m<<"\n"; }};
#define CHECK(x) do{ auto r=(x); if(r!=0){ std::cerr<<"CUDA error:"<<cudaGetErrorString(r)<<"\n"; std::abort();} }while(0)

static std::vector<char> readAll(const std::string& p){ std::ifstream f(p, std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0); std::vector<char> b(n); f.read(b.data(), n); return b; }

// Find first INPUT/OUTPUT name
static std::string firstByMode(const nvinfer1::ICudaEngine& e, nvinfer1::TensorIOMode mode){
	int n=e.getNbIOTensors();
	for(int i=0;i<n;++i){ const char* name=e.getIOTensorName(i); if(e.getTensorIOMode(name)==mode) return std::string(name); }
	return std::string();
}

struct LetterboxInfo{ float scale; int padX; int padY; int outW; int outH; };
static cv::Mat letterbox(const cv::Mat& img, int dstW, int dstH, LetterboxInfo& info){
	int iw=img.cols, ih=img.rows; float s = std::min(dstW/(float)iw, dstH/(float)ih);
	int nw = std::round(iw*s), nh = std::round(ih*s);
	int padX = (dstW - nw)/2, padY = (dstH - nh)/2;
	cv::Mat r; cv::resize(img, r, cv::Size(nw, nh));
	cv::Mat canvas(dstH, dstW, img.type(), cv::Scalar(114,114,114));
	r.copyTo(canvas(cv::Rect(padX, padY, nw, nh)));
	info = {s, padX, padY, dstW, dstH};
	return canvas;
}

static float iou(const cv::Rect2f& a, const cv::Rect2f& b){
	float inter = (a & b).area(); float uni = a.area()+b.area()-inter; return uni>0? inter/uni: 0.f;
}

struct Det{ cv::Rect2f box; int cls; float conf; };
static std::vector<int> nms(const std::vector<Det>& dets, float iouTh){
	std::vector<int> order(dets.size()); std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a,int b){return dets[a].conf>dets[b].conf;});
	std::vector<int> keep; std::vector<char> removed(dets.size(),0);
	for(size_t i=0;i<order.size();++i){ int idx=order[i]; if(removed[idx]) continue; keep.push_back(idx);
		for(size_t j=i+1;j<order.size();++j){ int idx2=order[j]; if(removed[idx2]) continue; if(iou(dets[idx].box, dets[idx2].box)>iouTh) removed[idx2]=1; }
	}
	return keep;
}

int main(int argc, char** argv){
	if(argc<2){
		std::cerr<<"Usage: "<<argv[0]<<" <engine.trt> [--image path] [--H 640] [--W 640] [--conf 0.25] [--iou 0.5]\n";
		return 1;
	}
	std::string eng=argv[1]; std::string imagePath; int H=640,W=640; float confTh=0.25f, iouTh=0.5f; int B=1;
	for(int i=2;i<argc;++i){ std::string t=argv[i];
		if(t=="--image" && i+1<argc) imagePath=argv[++i];
		else if(t=="--H" && i+1<argc) H=std::stoi(argv[++i]);
		else if(t=="--W" && i+1<argc) W=std::stoi(argv[++i]);
		else if(t=="--conf" && i+1<argc) confTh=std::stof(argv[++i]);
		else if(t=="--iou" && i+1<argc) iouTh=std::stof(argv[++i]);
	}

	Logger g;
	auto data = readAll(eng);
	auto rt = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g));
	auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(rt->deserializeCudaEngine(data.data(), data.size()));
	auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

	std::string inName = firstByMode(*engine, nvinfer1::TensorIOMode::kINPUT);
	std::string outName = firstByMode(*engine, nvinfer1::TensorIOMode::kOUTPUT);
	if(inName.empty()||outName.empty()){ std::cerr<<"Failed to find IO tensors\n"; return 2; }

	auto inShape = engine->getTensorShape(inName.c_str());
	inShape.d[0]=B; inShape.d[2]=H; inShape.d[3]=W;
	if(!ctx->setInputShape(inName.c_str(), inShape)){ std::cerr<<"setInputShape failed\n"; return 2; }

	size_t inSize = 1; for(int i=0;i<inShape.nbDims;++i) inSize*=inShape.d[i];
	auto outShape = ctx->getTensorShape(outName.c_str()); size_t outSize=1; for(int i=0;i<outShape.nbDims;++i) outSize*=std::max(1, outShape.d[i]);

	void *dIn=nullptr,*dOut=nullptr; CHECK(cudaMalloc(&dIn, inSize*sizeof(float))); CHECK(cudaMalloc(&dOut, outSize*sizeof(float)));
	ctx->setTensorAddress(inName.c_str(), dIn); ctx->setTensorAddress(outName.c_str(), dOut);

	std::vector<float> hIn(inSize, 0.5f), hOut(outSize);
	cv::Mat src;
	LetterboxInfo lb{};
	if(!imagePath.empty()){
		src = cv::imread(imagePath);
		if(src.empty()){ std::cerr<<"Failed to read image: "<<imagePath<<"\n"; return 3; }
		cv::Mat lbimg = letterbox(src, W, H, lb);
		cv::Mat rgb; cv::cvtColor(lbimg, rgb, cv::COLOR_BGR2RGB);
		cv::Mat f; rgb.convertTo(f, CV_32FC3, 1.0/255.0);
		std::vector<cv::Mat> ch; cv::split(f, ch);
		size_t plane = (size_t)W*H;
		// CHW
		memcpy(hIn.data()+0*plane, ch[0].data, plane*sizeof(float));
		memcpy(hIn.data()+1*plane, ch[1].data, plane*sizeof(float));
		memcpy(hIn.data()+2*plane, ch[2].data, plane*sizeof(float));
	}

	CHECK(cudaMemcpy(dIn, hIn.data(), inSize*sizeof(float), cudaMemcpyHostToDevice));
	cudaStream_t s; CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
	ctx->enqueueV3(s); CHECK(cudaMemcpyAsync(hOut.data(), dOut, outSize*sizeof(float), cudaMemcpyDeviceToHost, s)); CHECK(cudaStreamSynchronize(s));

	// Try to decode YOLOv8 head: support layouts [B,N,C], [B,C,N], [N,C], [C,N]
	if(!imagePath.empty()){
		int nbDims = outShape.nbDims;
		int N=0, C=0; bool layout_NC=true; // true if [..., N, C]
		if(nbDims==3){
			// Prefer [B,N,C] if last dim is small (C) and middle is large (N)
			int d0 = outShape.d[0], d1 = outShape.d[1], d2 = outShape.d[2];
			if(d1> d2){ N=d1; C=d2; layout_NC=true; }
			else { N=d2; C=d1; layout_NC=false; }
		} else if(nbDims==2){
			int d0 = outShape.d[0], d1 = outShape.d[1];
			if(d0> d1){ N=d0; C=d1; layout_NC=true; }
			else { N=d1; C=d0; layout_NC=false; }
		}
		std::vector<Det> dets;
		if(N>0 && C>=6){
			const float* p = hOut.data();
			auto get_val = [&](int i, int k){
				// i in [0,N), k in [0,C)
				if(nbDims==3){
					if(layout_NC) return p + i*C + k; // [B,N,C]
					else return p + k*N + i;          // [B,C,N]
				} else { // nbDims==2
					if(layout_NC) return p + i*C + k; // [N,C]
					else return p + k*N + i;          // [C,N]
				}
			};
			for(int i=0;i<N;++i){
				// If C==6 and row is [x1,y1,x2,y2,score,cls] (post-NMS), draw directly
				if(C==6){
					float x1 = *get_val(i,0), y1=*get_val(i,1), x2=*get_val(i,2), y2=*get_val(i,3);
					float conf = *get_val(i,4); int cls = (int)std::round(*get_val(i,5));
					if(conf<confTh) continue;
					// coords possibly already in original scale; if not, attempt reverse letterbox assuming xyxy in net space
					// Heuristic: if x2<=W+2 && y2<=H+2, then it's likely in net space
					if(x2 <= W+2 && y2 <= H+2){
						float nx1 = (x1 - lb.padX)/lb.scale;
						float ny1 = (y1 - lb.padY)/lb.scale;
						float nx2 = (x2 - lb.padX)/lb.scale;
						float ny2 = (y2 - lb.padY)/lb.scale;
						x1 = std::max(0.f, nx1); y1=std::max(0.f, ny1);
						x2 = std::min((float)src.cols-1, nx2); y2=std::min((float)src.rows-1, ny2);
					}
					dets.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), cls, conf});
					continue;
				}

				// Assume [x,y,w,h,(obj),class scores...]
				float x=*get_val(i,0), y=*get_val(i,1), w=*get_val(i,2), h=*get_val(i,3);
				int cls=-1; float clsScore=0.f, obj=1.f; int clsStart=4;
				if(C>=85){ obj = *get_val(i,4); clsStart=5; }
				for(int k=clsStart;k<C;++k){ float v=*get_val(i,k); if(v>clsScore){ clsScore=v; cls=k-clsStart; } }
				float conf = obj*clsScore; if(conf<confTh) continue;
				// xywh (net) -> xyxy (orig)
				float bx = (x - lb.padX)/lb.scale;
				float by = (y - lb.padY)/lb.scale;
				float bw = w / lb.scale; float bh = h / lb.scale;
				float x1 = std::max(0.f, bx - bw/2.f);
				float y1 = std::max(0.f, by - bh/2.f);
				float x2 = std::min((float)src.cols-1, bx + bw/2.f);
				float y2 = std::min((float)src.rows-1, by + bh/2.f);
				dets.push_back({cv::Rect2f(x1,y1,x2-x1,y2-y1), cls, conf});
			}
			auto keep = nms(dets, iouTh);
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

	CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut)); CHECK(cudaStreamDestroy(s));
	return 0;
}
