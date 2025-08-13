#include <iostream>
#include <fstream>
#include <memory>
#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include <trt_utils/trt_builder.h>
#include <trt_utils/trt_common.h>

// forward decl from calibrator cpp
std::unique_ptr<nvinfer1::IInt8Calibrator> createYoloCalibrator(int,int,int,const std::string&,const std::string&);

int main(int argc, char** argv){
	if(argc<3){
		std::cerr << "Usage: "<<argv[0]<<" <model.onnx> <output.trt|prefix> [--fp32|--fp16|--int8] [--calib-dir DIR] [--min SHAPE] [--opt SHAPE] [--max SHAPE]\n";
		std::cerr << "  SHAPE supports HxW (e.g., 640x640) or NxCxHxW (e.g., 1x3x640x640). Batch is taken from --max if provided.\n";
		return 1;
	}

	std::string onnx = argv[1];
	std::string outArg = argv[2];
	std::string precisionFlag; // fp32|fp16|int8
	std::string calibDir;      // from --calib-dir or env
	// Defaults for YOLO
	int minH=0,minW=0,optH=0,optW=0,maxH=0,maxW=0; // 0 means use ONNX/default
	int maxBatchFromShape = -1; // -1 means keep default

	auto toLower=[](std::string s){ std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){return (char)std::tolower(c);}); return s; };

	auto parseShape = [&](const std::string& s, int& H, int& W, int* outBatch){
		// Accept HxW or NxCxHxW; take last two tokens as H, W; optionally first as batch
		std::vector<int> vals; vals.reserve(4);
		std::string cur; for(char c: s){ if(c=='x' || c=='X'){ if(!cur.empty()){ vals.push_back(std::atoi(cur.c_str())); cur.clear(); } } else cur.push_back(c); }
		if(!cur.empty()) vals.push_back(std::atoi(cur.c_str()));
		if(vals.size()>=2){ H = vals[vals.size()-2]; W = vals[vals.size()-1]; if(outBatch && vals.size()>=4) *outBatch = vals[0]; return true; }
		return false;
	};

	// Backward-compat: third positional precision
	if(argc>=4 && argv[3][0] != '-'){
		precisionFlag = toLower(argv[3]);
		if(argc>=5) calibDir = argv[4];
		if(argc>=6){ parseShape(argv[5], minH, minW, nullptr); }
		if(argc>=7){ parseShape(argv[6], optH, optW, nullptr); }
		if(argc>=8){ parseShape(argv[7], maxH, maxW, &maxBatchFromShape); }
	}

	// Parse named flags
	for(int i=3;i<argc;++i){
		std::string t = argv[i];
		if(t=="--fp32") precisionFlag = "fp32";
		else if(t=="--fp16") precisionFlag = "fp16";
		else if(t=="--int8") precisionFlag = "int8";
		else if(t=="--calib-dir" && i+1<argc) { calibDir = argv[++i]; }
		else if(t=="--min" && i+1<argc) { parseShape(argv[++i], minH, minW, nullptr); }
		else if(t=="--opt" && i+1<argc) { parseShape(argv[++i], optH, optW, nullptr); }
		else if(t=="--max" && i+1<argc) { parseShape(argv[++i], maxH, maxW, &maxBatchFromShape); }
	}

	if(precisionFlag.empty()){
		// Default to fp16 if not provided
		precisionFlag = "fp16";
	}

	if(calibDir.empty()){
		if(const char* e = std::getenv("CALIB_DATA_DIR")) calibDir = e;
	}

	TrtLogger g;
	TrtEngineBuilder teb(g);
	BuildOptions opt;
	opt.precision = precisionFlag; // fp32|fp16|int8
	opt.calibDataDir = calibDir;   // used when int8
	if(maxBatchFromShape>0) opt.maxBatch = maxBatchFromShape; else opt.maxBatch = 32;
	// dynamic H/W
	opt.hwMinH = minH; opt.hwMinW = minW;
	opt.hwOptH = optH; opt.hwOptW = optW;
	opt.hwMaxH = maxH; opt.hwMaxW = maxW;

	int inW=0, inH=0; std::string inName;
	std::unique_ptr<nvinfer1::IInt8Calibrator> extCalib;
	if(precisionFlag=="int8"){
		if(calibDir.empty()){ std::cerr<<"INT8 requires --calib-dir or CALIB_DATA_DIR"<<"\n"; return 1; }
		// 创建一个可选的外部校准器；TrtEngineBuilder 会优先使用传入的校准器
		int calibW = (opt.hwOptW>0? opt.hwOptW : 640);
		int calibH = (opt.hwOptH>0? opt.hwOptH : 640);
		// Note: W/H here refer to desired network input size; batch fixed to 8 for calibration
		extCalib = createYoloCalibrator(8, calibW, calibH, calibDir, "yolo_int8.cache");
	}

	auto ser = teb.buildFromOnnx(onnx, opt, inW, inH, inName, extCalib.get());
	if(!ser){ std::cerr<<"Build failed\n"; return 2; }
	// If output argument ends with .trt, use it directly; else append _<prec>.trt
	auto endsWith = [](const std::string& s, const std::string& suf){ return s.size()>=suf.size() && s.compare(s.size()-suf.size(), suf.size(), suf)==0; };
	std::string out = outArg;
	if(!endsWith(outArg, ".trt")){
		out = outArg + (precisionFlag=="fp32"? "" : ("_"+precisionFlag)) + ".trt";
	}
	if(!EngineIO::writeFile(out, ser->data(), ser->size())){ std::cerr<<"Write failed: "<<out<<"\n"; return 2; }
	std::cout<<"Saved: "<<out<<"\n";
	return 0;
}
