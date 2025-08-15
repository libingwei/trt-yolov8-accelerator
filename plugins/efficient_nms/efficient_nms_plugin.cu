#include "efficient_nms_plugin.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

namespace YoloTrt {

// CUDA device functions for NMS
__device__ float compute_iou(const float* box1, const float* box2) {
    float x1_min = fminf(box1[0], box1[2]);
    float y1_min = fminf(box1[1], box1[3]);
    float x1_max = fmaxf(box1[0], box1[2]);
    float y1_max = fmaxf(box1[1], box1[3]);
    
    float x2_min = fminf(box2[0], box2[2]);
    float y2_min = fminf(box2[1], box2[3]);
    float x2_max = fmaxf(box2[0], box2[2]);
    float y2_max = fmaxf(box2[1], box2[3]);
    
    float inter_x1 = fmaxf(x1_min, x2_min);
    float inter_y1 = fmaxf(y1_min, y2_min);
    float inter_x2 = fminf(x1_max, x2_max);
    float inter_y2 = fminf(y1_max, y2_max);
    
    float inter_area = fmaxf(0.0f, inter_x2 - inter_x1) * fmaxf(0.0f, inter_y2 - inter_y1);
    float area1 = (x1_max - x1_min) * (y1_max - y1_min);
    float area2 = (x2_max - x2_min) * (y2_max - y2_min);
    float union_area = area1 + area2 - inter_area;
    
    return (union_area > 0.0f) ? (inter_area / union_area) : 0.0f;
}

__global__ void nms_kernel(const float* boxes, int numBoxes, float scoreThreshold, 
                          float iouThreshold, bool classAgnostic, bool* suppressed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numBoxes) return;
    
    // Check score threshold
    float score = boxes[idx * 6 + 4]; // confidence
    if (score < scoreThreshold) {
        suppressed[idx] = true;
        return;
    }
    
    // Check against higher scoring boxes
    for (int i = 0; i < idx; i++) {
        if (suppressed[i]) continue;
        
        float other_score = boxes[i * 6 + 4];
        if (other_score < score) continue; // Only check higher scoring boxes
        
        // Class-agnostic or same class check
        if (!classAgnostic) {
            int cls1 = (int)boxes[idx * 6 + 5];
            int cls2 = (int)boxes[i * 6 + 5];
            if (cls1 != cls2) continue;
        }
        
        // Compute IoU
        float iou = compute_iou(&boxes[idx * 6], &boxes[i * 6]);
        if (iou > iouThreshold) {
            suppressed[idx] = true;
            break;
        }
    }
}

__global__ void copy_filtered_boxes(const float* boxes, bool* suppressed, int numBoxes,
                                   int maxOutputBoxes, float* outputBoxes, int* outputCount) {
    int count = 0;
    for (int i = 0; i < numBoxes && count < maxOutputBoxes; i++) {
        if (!suppressed[i]) {
            for (int j = 0; j < 6; j++) {
                outputBoxes[count * 6 + j] = boxes[i * 6 + j];
            }
            count++;
        }
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *outputCount = count;
    }
}

// CUDA NMS kernel implementation
extern "C" void efficient_nms_kernel(
    const float* boxes, int numBoxes, int batchSize,
    float scoreThreshold, float iouThreshold, int maxOutputBoxes,
    bool classAgnostic, float* outputBoxes, int* outputNums, cudaStream_t stream) {
    
    // Allocate temporary memory for suppression flags
    bool* suppressed;
    cudaMalloc(&suppressed, numBoxes * sizeof(bool));
    cudaMemset(suppressed, 0, numBoxes * sizeof(bool));
    
    // Launch NMS kernel
    int blockSize = 256;
    int gridSize = (numBoxes + blockSize - 1) / blockSize;
    nms_kernel<<<gridSize, blockSize, 0, stream>>>(
        boxes, numBoxes, scoreThreshold, iouThreshold, classAgnostic, suppressed);
    
    // Copy filtered results
    copy_filtered_boxes<<<1, 1, 0, stream>>>(
        boxes, suppressed, numBoxes, maxOutputBoxes, outputBoxes, outputNums);
    
    cudaFree(suppressed);
}

// Plugin implementation
EfficientNmsPlugin::EfficientNmsPlugin(const std::string& name, float scoreThreshold, 
                                       float iouThreshold, int maxOutputBoxes, bool classAgnostic)
    : mLayerName(name), mScoreThreshold(scoreThreshold), mIouThreshold(iouThreshold),
      mMaxOutputBoxes(maxOutputBoxes), mClassAgnostic(classAgnostic), mBatchSize(1) {}

EfficientNmsPlugin::EfficientNmsPlugin(const std::string& name, const void* data, size_t length)
    : mLayerName(name) {
    const char* d = static_cast<const char*>(data);
    size_t offset = 0;
    
    memcpy(&mScoreThreshold, d + offset, sizeof(float)); offset += sizeof(float);
    memcpy(&mIouThreshold, d + offset, sizeof(float)); offset += sizeof(float);
    memcpy(&mMaxOutputBoxes, d + offset, sizeof(int)); offset += sizeof(int);
    memcpy(&mClassAgnostic, d + offset, sizeof(bool)); offset += sizeof(bool);
    memcpy(&mBatchSize, d + offset, sizeof(size_t)); offset += sizeof(size_t);
}

nvinfer1::IPluginV2DynamicExt* EfficientNmsPlugin::clone() const noexcept {
    auto plugin = new EfficientNmsPlugin(mLayerName, mScoreThreshold, mIouThreshold, 
                                         mMaxOutputBoxes, mClassAgnostic);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs EfficientNmsPlugin::getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
                                                           int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept {
    // Input: [N, 6] (x1,y1,x2,y2,score,class)
    // Output 0: [maxOutputBoxes, 6] (filtered boxes)
    // Output 1: [1] (number of valid boxes)
    
    nvinfer1::DimsExprs output;
    if (outputIndex == 0) {
        output.nbDims = 2;
        output.d[0] = exprBuilder.constant(mMaxOutputBoxes);
        output.d[1] = exprBuilder.constant(6);
    } else if (outputIndex == 1) {
        output.nbDims = 1;
        output.d[0] = exprBuilder.constant(1);
    }
    return output;
}

bool EfficientNmsPlugin::supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, 
                                                   int nbInputs, int nbOutputs) noexcept {
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT && 
            inOut[pos].format == nvinfer1::TensorFormat::kLINEAR);
}

void EfficientNmsPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                                        const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept {
    // Store configuration if needed
}

size_t EfficientNmsPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                                           const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept {
    // Return workspace needed for NMS computation
    return 0; // Simplified implementation
}

int EfficientNmsPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDescs, 
                                const nvinfer1::PluginTensorDesc* outputDescs,
                                const void* const* inputs, void* const* outputs, 
                                void* workspace, cudaStream_t stream) noexcept {
    
    const auto& inDesc = inputDescs[0];
    int N = inDesc.dims.d[0]; // Number of boxes
    int batchSize = 1; // Simplified for single batch
    
    const float* inputBoxes = static_cast<const float*>(inputs[0]);
    float* outputBoxes = static_cast<float*>(outputs[0]);
    int* outputNums = static_cast<int*>(outputs[1]);
    
    // Call CUDA NMS kernel
    efficient_nms_kernel(inputBoxes, N, batchSize, mScoreThreshold, mIouThreshold, 
                        mMaxOutputBoxes, mClassAgnostic, outputBoxes, outputNums, stream);
    
    return 0;
}

nvinfer1::DataType EfficientNmsPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
                                                         int nbInputs) const noexcept {
    if (index == 0) return nvinfer1::DataType::kFLOAT; // boxes
    else return nvinfer1::DataType::kINT32; // counts
}

const char* EfficientNmsPlugin::getPluginType() const noexcept {
    return "EfficientNMS_TRT";
}

const char* EfficientNmsPlugin::getPluginVersion() const noexcept {
    return "1";
}

int EfficientNmsPlugin::getNbOutputs() const noexcept {
    return 2; // boxes + counts
}

int EfficientNmsPlugin::initialize() noexcept {
    return 0;
}

void EfficientNmsPlugin::terminate() noexcept {}

size_t EfficientNmsPlugin::getSerializationSize() const noexcept {
    return sizeof(float) * 2 + sizeof(int) + sizeof(bool) + sizeof(size_t);
}

void EfficientNmsPlugin::serialize(void* buffer) const noexcept {
    char* d = static_cast<char*>(buffer);
    size_t offset = 0;
    
    memcpy(d + offset, &mScoreThreshold, sizeof(float)); offset += sizeof(float);
    memcpy(d + offset, &mIouThreshold, sizeof(float)); offset += sizeof(float);
    memcpy(d + offset, &mMaxOutputBoxes, sizeof(int)); offset += sizeof(int);
    memcpy(d + offset, &mClassAgnostic, sizeof(bool)); offset += sizeof(bool);
    memcpy(d + offset, &mBatchSize, sizeof(size_t)); offset += sizeof(size_t);
}

void EfficientNmsPlugin::destroy() noexcept {
    delete this;
}

void EfficientNmsPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* EfficientNmsPlugin::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// Plugin Creator implementation
nvinfer1::PluginFieldCollection EfficientNmsPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> EfficientNmsPluginCreator::mPluginAttributes;

EfficientNmsPluginCreator::EfficientNmsPluginCreator() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("score_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("iou_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("class_agnostic", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* EfficientNmsPluginCreator::getPluginName() const noexcept {
    return "EfficientNMS_TRT";
}

const char* EfficientNmsPluginCreator::getPluginVersion() const noexcept {
    return "1";
}

const nvinfer1::PluginFieldCollection* EfficientNmsPluginCreator::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2* EfficientNmsPluginCreator::createPlugin(const char* name, 
                                                           const nvinfer1::PluginFieldCollection* fc) noexcept {
    float scoreThreshold = 0.25f;
    float iouThreshold = 0.5f;
    int maxOutputBoxes = 100;
    bool classAgnostic = true;
    
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fc->fields[i].name;
        if (!strcmp(attrName, "score_threshold")) {
            scoreThreshold = *(static_cast<const float*>(fc->fields[i].data));
        } else if (!strcmp(attrName, "iou_threshold")) {
            iouThreshold = *(static_cast<const float*>(fc->fields[i].data));
        } else if (!strcmp(attrName, "max_output_boxes")) {
            maxOutputBoxes = *(static_cast<const int*>(fc->fields[i].data));
        } else if (!strcmp(attrName, "class_agnostic")) {
            classAgnostic = *(static_cast<const int*>(fc->fields[i].data)) != 0;
        }
    }
    
    auto plugin = new EfficientNmsPlugin(name, scoreThreshold, iouThreshold, maxOutputBoxes, classAgnostic);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::IPluginV2* EfficientNmsPluginCreator::deserializePlugin(const char* name, 
                                                                 const void* serialData, 
                                                                 size_t serialLength) noexcept {
    auto plugin = new EfficientNmsPlugin(name, serialData, serialLength);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

void EfficientNmsPluginCreator::setPluginNamespace(const char* pluginNamespace) noexcept {
    mNamespace = pluginNamespace;
}

const char* EfficientNmsPluginCreator::getPluginNamespace() const noexcept {
    return mNamespace.c_str();
}

// Plugin registration function
extern "C" void registerEfficientNmsPlugin(){
    static EfficientNmsPluginCreator creator;
    auto* reg = getPluginRegistry();
    if(reg) { 
        // Register with empty namespace (default)
        reg->registerCreator(creator, "");
        // Optionally set a custom namespace if needed
        creator.setPluginNamespace(""); 
    }
}

} // namespace YoloTrt
