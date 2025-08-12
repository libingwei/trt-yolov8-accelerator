#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cstdlib>

#include <trt_utils/trt_builder.h>
#include <trt_utils/trt_common.h>

// forward decl from calibrator cpp
std::unique_ptr<nvinfer1::IInt8Calibrator> createYoloCalibrator(int,int,int,const std::string&,const std::string&);

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

	TrtLogger g;
	TrtEngineBuilder teb(g);
	BuildOptions opt;
	opt.precision = prec; // fp32|fp16|int8
	opt.calibDataDir = calibDir; // used when int8
	opt.maxBatch = 32;

	int inW=0, inH=0; std::string inName;
	std::unique_ptr<nvinfer1::IInt8Calibrator> extCalib;
	if(prec=="int8"){
		if(calibDir.empty()){ std::cerr<<"INT8 requires calib_dir"<<"\n"; return 1; }
		// 创建一个可选的外部校准器；TrtEngineBuilder 会优先使用传入的校准器
		extCalib = createYoloCalibrator(8, optW, optH, calibDir, "yolo_int8.cache");
	}

	auto ser = teb.buildFromOnnx(onnx, opt, inW, inH, inName, extCalib.get());
	if(!ser){ std::cerr<<"Build failed\n"; return 2; }
	std::string out = outp + (prec=="fp32"? "": ("_"+prec)) + ".trt";
	if(!EngineIO::writeFile(out, ser->data(), ser->size())){ std::cerr<<"Write failed: "<<out<<"\n"; return 2; }
	std::cout<<"Saved: "<<out<<"\n";
	return 0;
}
