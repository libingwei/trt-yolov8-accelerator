#pragma once
#include "NvInfer.h"
#include <string>
#include <vector>

// Minimal skeleton for a YOLO decode plugin (IPluginV2DynamicExt)
// This is a placeholder for future implementation; currently unused by default.

class DecodeYoloPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    DecodeYoloPlugin(const std::string& name, bool useLetterbox = false) 
        : mLayerName(name), mUseLetterbox(useLetterbox) {}
    DecodeYoloPlugin(const void* data, size_t length) : mUseLetterbox(false) { 
        (void)data; (void)length; 
        // TODO: deserialize mUseLetterbox from data
    }

    // IPluginV2
    const char* getPluginType() const noexcept override { return "DecodeYolo_TRT"; }
    const char* getPluginVersion() const noexcept override { return "2"; }
    int getNbOutputs() const noexcept override { return 1; }
    int initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return sizeof(bool); }
    void serialize(void* buffer) const noexcept override { 
        *static_cast<bool*>(buffer) = mUseLetterbox; 
    }
    void destroy() noexcept override { delete this; }
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override { 
        return new DecodeYoloPlugin(mLayerName, mUseLetterbox); 
    }
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int, const nvinfer1::DataType* inputTypes, int) const noexcept override {
        return inputTypes[0];
    }

    // IPluginV2DynamicExt
    nvinfer1::DimsExprs getOutputDimensions(int, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder&) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        // Expect: input 0 = YOLO head [N,C], optional input 1 = letterbox params [N,4] (padX, padY, scale, origScale)
        int totalIO = nbInputs + nbOutputs;
        if (pos >= totalIO) return false;
        
        if (mUseLetterbox && nbInputs >= 2) {
            if (pos == 0 || pos == 2) // YOLO head or output
                return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
            else if (pos == 1) // letterbox params
                return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
        } else {
            if (pos < nbInputs || pos == nbInputs) // inputs or output
                return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR && inOut[pos].type == nvinfer1::DataType::kFLOAT;
        }
        return false;
    }
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc*, int, const nvinfer1::DynamicPluginTensorDesc*, int) noexcept override {}
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc*, int, const nvinfer1::PluginTensorDesc*, int) const noexcept override { return 0; }
    int enqueue(const nvinfer1::PluginTensorDesc*, const nvinfer1::PluginTensorDesc*, const void* const*, void* const*, void*, cudaStream_t) noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;
    bool mUseLetterbox;
};

class DecodeYoloPluginCreator : public nvinfer1::IPluginCreator {
public:
    const char* getPluginName() const noexcept override { return "DecodeYolo_TRT"; }
    const char* getPluginVersion() const noexcept override { return "2"; }
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection*) noexcept override { return new DecodeYoloPlugin(name); }
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override { return new DecodeYoloPlugin(serialData, serialLength); }
    void setPluginNamespace(const char* ns) noexcept override { mNamespace = ns ? ns : ""; }
    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }
private:
    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{0, nullptr};
};

// Helper to register plugin creator into TensorRT registry (must be called before engine deserialization)
extern "C" void registerDecodeYoloPlugin();
