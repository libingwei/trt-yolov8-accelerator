#pragma once
#include <NvInfer.h>
#include <memory>
#include <string>
#include <vector>

namespace YoloTrt {

class EfficientNmsPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    EfficientNmsPlugin(const std::string& name, float scoreThreshold = 0.25f, 
                       float iouThreshold = 0.5f, int maxOutputBoxes = 100,
                       bool classAgnostic = true);
    
    EfficientNmsPlugin(const std::string& name, const void* data, size_t length);
    
    ~EfficientNmsPlugin() override = default;

    // IPluginV2DynamicExt methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(int outputIndex, const nvinfer1::DimsExprs* inputs, 
                                            int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, 
                                   int nbInputs, int nbOutputs) noexcept override;
    void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                         const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                            const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
    int enqueue(const nvinfer1::PluginTensorDesc* inputDescs, const nvinfer1::PluginTensorDesc* outputDescs,
                const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override;

    // IPluginV2 methods
    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    int getNbOutputs() const noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mLayerName;
    std::string mNamespace;
    float mScoreThreshold;
    float mIouThreshold;
    int mMaxOutputBoxes;
    bool mClassAgnostic;
    size_t mBatchSize;
};

class EfficientNmsPluginCreator : public nvinfer1::IPluginCreator {
public:
    EfficientNmsPluginCreator();
    ~EfficientNmsPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

// Plugin registration function
extern "C" void registerEfficientNmsPlugin();

} // namespace YoloTrt
