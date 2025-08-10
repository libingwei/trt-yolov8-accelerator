#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <memory>
#include "NvInfer.h"
#include "cuda_runtime_api.h"

class Logger : public nvinfer1::ILogger{ void log(Severity s, const char* m) noexcept override { if(s<=Severity::kWARNING) std::cout<<m<<"\n"; }};
#define CHECK(x) do{ auto r=(x); if(r!=0){ std::cerr<<"CUDA error:"<<cudaGetErrorString(r)<<"\n"; std::abort();} }while(0)

static std::vector<char> readAll(const std::string& p){ std::ifstream f(p, std::ios::binary); f.seekg(0,std::ios::end); size_t n=f.tellg(); f.seekg(0); std::vector<char> b(n); f.read(b.data(), n); return b; }

int main(int argc, char** argv){
	if(argc<2){ std::cerr<<"Usage: "<<argv[0]<<" <engine.trt> [H] [W] [batch]\n"; return 1; }
	std::string eng=argv[1]; int H=640,W=640,B=1; if(argc>=3) H=std::stoi(argv[2]); if(argc>=4) W=std::stoi(argv[3]); if(argc>=5) B=std::stoi(argv[4]);
	Logger g;
	auto data = readAll(eng);
	auto rt = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g));
	auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(rt->deserializeCudaEngine(data.data(), data.size()));
	auto ctx = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

	// assume first input/output
	const char* inName = engine->getIOTensorName(0); const char* outName = engine->getIOTensorName(1);
	auto inShape = engine->getTensorShape(inName); inShape.d[0]=B; inShape.d[2]=H; inShape.d[3]=W;
	if(!ctx->setInputShape(inName, inShape)){ std::cerr<<"setInputShape failed\n"; return 2; }

	size_t inSize = 1; for(int i=0;i<inShape.nbDims;++i) inSize*=inShape.d[i];
	auto outShape = ctx->getTensorShape(outName); size_t outSize=1; for(int i=0;i<outShape.nbDims;++i) outSize*=outShape.d[i];

	void *dIn=nullptr,*dOut=nullptr; CHECK(cudaMalloc(&dIn, inSize*sizeof(float))); CHECK(cudaMalloc(&dOut, outSize*sizeof(float)));
	ctx->setTensorAddress(inName, dIn); ctx->setTensorAddress(outName, dOut);
	std::vector<float> hIn(inSize, 0.5f), hOut(outSize);
	CHECK(cudaMemcpy(dIn, hIn.data(), inSize*sizeof(float), cudaMemcpyHostToDevice));
	cudaStream_t s; CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
	ctx->enqueueV3(s); CHECK(cudaMemcpyAsync(hOut.data(), dOut, outSize*sizeof(float), cudaMemcpyDeviceToHost, s)); CHECK(cudaStreamSynchronize(s));
	std::cout<<"Ran once. in="<<inSize<<" out="<<outSize<<"\n";
	CHECK(cudaFree(dIn)); CHECK(cudaFree(dOut)); CHECK(cudaStreamDestroy(s));
	return 0;
}
