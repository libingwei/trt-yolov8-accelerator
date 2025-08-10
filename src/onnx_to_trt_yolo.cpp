#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cstdlib>
#include "NvInfer.h"
#include "NvOnnxParser.h"

// forward decl from calibrator cpp
std::unique_ptr<nvinfer1::IInt8Calibrator> createYoloCalibrator(int,int,int,const std::string&,const std::string&);

class Logger : public nvinfer1::ILogger{ void log(Severity s, const char* m) noexcept override { if(s<=Severity::kWARNING) std::cout<<m<<"\n"; }};

int main(int argc, char** argv){
	if(argc<4){
		std::cerr << "Usage: "<<argv[0]<<" <model.onnx> <out_prefix> <fp32|fp16|int8> [calib_dir] [minHW] [optHW] [maxHW]\n";
		return 1;
	}
	std::string onnx=argv[1], outp=argv[2], prec=argv[3];
	std::string calibDir = (argc>=5? argv[4]: (std::getenv("CALIB_DATA_DIR")? std::getenv("CALIB_DATA_DIR"): ""));
	auto parseHW=[&](const char* s,int& H,int& W){ if(!s) return false; return sscanf(s, "%dx%d", &H,&W)==2; };
	int minH=320,minW=320,optH=640,optW=640,maxH=1280,maxW=1280;
	if(argc>=6) parseHW(argv[5],minH,minW); if(argc>=7) parseHW(argv[6],optH,optW); if(argc>=8) parseHW(argv[7],maxH,maxW);

	Logger g;
	auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g));
	auto config  = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
	auto parser  = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, g));
	if(!parser->parseFromFile(onnx.c_str(), (int)Logger::Severity::kINFO)){
		std::cerr<<"ONNX parse failed: "<<onnx<<"\n"; return 1; }

	if(prec=="fp16") config->setFlag(nvinfer1::BuilderFlag::kFP16);
	if(prec=="int8") config->setFlag(nvinfer1::BuilderFlag::kINT8);

	// build profile: assume input 0 is NCHW
	auto in0 = network->getInput(0); auto dims = in0->getDimensions();
	// N,C,H,W with -1 for dynamic
	dims.d[0] = -1; dims.d[2] = -1; dims.d[3] = -1; in0->setDimensions(dims);
	auto profile = builder->createOptimizationProfile();
	profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1,dims.d[1], minH, minW});
	profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{1,dims.d[1], optH, optW});
	profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{32,dims.d[1], maxH, maxW});
	config->addOptimizationProfile(profile);

	std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
	if(config->getFlag(nvinfer1::BuilderFlag::kINT8)){
		if(calibDir.empty()){ std::cerr<<"INT8 requires calib_dir"<<"\n"; return 1; }
		calibrator = createYoloCalibrator(8, optW, optH, calibDir, "yolo_int8.cache");
		config->setInt8Calibrator(calibrator.get());
	}

	std::cout<<"Building engine..."<<std::endl;
	auto ser = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
	if(!ser){ std::cerr<<"Build failed\n"; return 2; }
	std::string out = outp + (prec=="fp32"? "": ("_"+prec)) + ".trt";
	std::ofstream f(out, std::ios::binary); f.write((const char*)ser->data(), ser->size());
	std::cout<<"Saved: "<<out<<"\n";
	return 0;
}
