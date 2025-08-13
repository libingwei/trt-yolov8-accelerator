#include "decode_yolo_plugin.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

// Simple CUDA kernel: convert [N, C] raw head [x,y,w,h,obj,cls...] or [x1,y1,x2,y2,conf,cls]
// into [N,6] with [x1,y1,x2,y2,conf,cls] without NMS. This mirrors the CPU decode in trt_utils.
// Note: This kernel assumes a single output tensor and does not apply letterbox reverse mapping.
// Mapping back to original image space should be done outside if needed (requires pad/scale per-sample).

namespace {
__device__ __forceinline__ float clampf(float v, float lo, float hi){ return fminf(fmaxf(v, lo), hi); }

__global__ void decode_kernel(const float* __restrict__ in, int N, int C,
                              int hasObj, int numClasses, int coordsIsXYWH,
                              float confTh,
                              float* __restrict__ out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i>=N) return;
    const float* p = in + i*C;
    float x = p[0], y = p[1], w = p[2], h = p[3];
    float obj = hasObj ? p[4] : 1.f;
    int clsBest = 0; float clsScore = 0.f;
    int clsStart = hasObj ? 5 : 4;
    for(int k=0;k<numClasses && (clsStart+k)<C; ++k){ float v = p[clsStart+k]; if(v>clsScore){ clsScore=v; clsBest=k; } }
    float conf = obj * clsScore;
    if(conf < confTh){ out[i*6+0]=0; out[i*6+1]=0; out[i*6+2]=0; out[i*6+3]=0; out[i*6+4]=0; out[i*6+5]=-1; return; }
    float x1, y1, x2, y2;
    if(coordsIsXYWH){
        x1 = x - w*0.5f; y1 = y - h*0.5f; x2 = x + w*0.5f; y2 = y + h*0.5f;
    } else {
        x1 = x; y1 = y; x2 = w; y2 = h;
    }
    out[i*6+0] = x1; out[i*6+1] = y1; out[i*6+2] = x2; out[i*6+3] = y2; out[i*6+4] = conf; out[i*6+5] = (float)clsBest;
}
}

using namespace nvinfer1;

nvinfer1::DimsExprs DecodeYoloPlugin::getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& eb) noexcept {
    // Expect [N, C] or [B, N, C] flattened to 2D in TRT; here we simply return [N,6]
    DimsExprs out;
    out.nbDims = 2;
    if(nbInputs<1){ out.d[0] = eb.constant(0); out.d[1] = eb.constant(6); return out; }
    const auto& in = inputs[0];
    if(in.nbDims==2){ out.d[0] = in.d[0]; }
    else if(in.nbDims==3){ out.d[0] = in.d[1]; }
    else { out.d[0] = eb.constant(0); }
    out.d[1] = eb.constant(6);
    return out;
}

int DecodeYoloPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDescs,
                              const nvinfer1::PluginTensorDesc* outputDescs,
                              const void* const* inputs,
                              void* const* outputs,
                              void*, cudaStream_t stream) noexcept {
    const auto& inDesc = inputDescs[0];
    int nbDims = inDesc.dims.nbDims;
    int N = 0, C = 0;
    if(nbDims==2){ N = inDesc.dims.d[0]; C = inDesc.dims.d[1]; }
    else if(nbDims==3){ N = inDesc.dims.d[1]; C = inDesc.dims.d[2]; }
    else { return 1; }

    const float* in = static_cast<const float*>(inputs[0]);
    float* out = static_cast<float*>(outputs[0]);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    decode_kernel<<<blocks, threads, 0, stream>>>(in, N, C, /*hasObj*/1, /*numClasses*/(C-5), /*coordsIsXYWH*/1, /*confTh*/0.0f, out);
    return 0;
}

extern "C" void registerDecodeYoloPlugin(){
    static DecodeYoloPluginCreator creator;
    auto* reg = getPluginRegistry();
    if(reg){ reg->registerCreator(creator, ""); }
}
