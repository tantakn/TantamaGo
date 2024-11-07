#include <iostream>
#include <fstream>
// #include <NvInfer.h>
#include "/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include/NvInfer.h"
#include "/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include/NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

using namespace nvinfer1;

// Loggerの実装
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
};

std::string execCommand(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::shared_ptr<FILE> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() に失敗しました！");
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

int main() {
    // Loggerの初期化
    Logger logger;

    // Runtimeの作成
    IRuntime* runtime = createInferRuntime(logger);

    // ONNXモデルからEngineの作成
    ICudaEngine* engine = nullptr;
    {
        // Builderの作成
        IBuilder* builder = createInferBuilder(logger);
        // Networkの作成
        INetworkDefinition* network = builder->createNetworkV2(0);
        // ONNXパーサーの作成
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        // ONNXモデルの読み込み
        parser->parseFromFile("test.onnx", static_cast<int>(ILogger::Severity::kWARNING));
        // エンジンのビルド
        IBuilderConfig* config = builder->createBuilderConfig();
        config->setMaxWorkspaceSize(1 << 20);  // 1MB
        engine = builder->buildEngineWithConfig(*network, *config);
        // リソースの解放
        parser->destroy();
        network->destroy();
        builder->destroy();
        config->destroy();
    }

    // Contextの作成
    IExecutionContext* context = engine->createExecutionContext();

    // 入力データの準備（バッチサイズ1、チャンネル6、サイズ9x9）
    int batchSize = 1;
    int inputIndex = engine->getBindingIndex("input");
    Dims inputDims = engine->getBindingDimensions(inputIndex);
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i)
        inputSize *= inputDims.d[i];
    float* inputData = new float[inputSize];
    // 入力データを設定（例として1.0で初期化）
    for (size_t i = 0; i < inputSize; ++i)
        inputData[i] = 1.0f;

    // 出力データの準備
    int outputPolicyIndex = engine->getBindingIndex("output_policy");
    int outputValueIndex = engine->getBindingIndex("output_value");
    Dims outputPolicyDims = engine->getBindingDimensions(outputPolicyIndex);
    size_t outputPolicySize = 1;
    for (int i = 0; i < outputPolicyDims.nbDims; ++i)
        outputPolicySize *= outputPolicyDims.d[i];
    float* outputPolicyData = new float[outputPolicySize];

    Dims outputValueDims = engine->getBindingDimensions(outputValueIndex);
    size_t outputValueSize = 1;
    for (int i = 0; i < outputValueDims.nbDims; ++i)
        outputValueSize *= outputValueDims.d[i];
    float* outputValueData = new float[outputValueSize];

    // デバイスメモリの確保
    void* buffers[3];
    cudaMalloc(&buffers[inputIndex], inputSize * sizeof(float));
    cudaMalloc(&buffers[outputPolicyIndex], outputPolicySize * sizeof(float));
    cudaMalloc(&buffers[outputValueIndex], outputValueSize * sizeof(float));

    // データをデバイスに転送
    cudaMemcpy(buffers[inputIndex], inputData, inputSize * sizeof(float), cudaMemcpyHostToDevice);

    // 推論の実行
    context->executeV2(buffers);

    // 結果をホストにコピー
    cudaMemcpy(outputPolicyData, buffers[outputPolicyIndex], outputPolicySize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(outputValueData, buffers[outputValueIndex], outputValueSize * sizeof(float), cudaMemcpyDeviceToHost);

    // 結果の表示
    std::cout << "Policy Output:" << std::endl;
    for (size_t i = 0; i < outputPolicySize; ++i)
        std::cout << outputPolicyData[i] << " ";
    std::cout << std::endl;

    std::cout << "Value Output:" << std::endl;
    for (size_t i = 0; i < outputValueSize; ++i)
        std::cout << outputValueData[i] << " ";
    std::cout << std::endl;

    // Pythonスクリプトの呼び出し
    std::string pythonScript = "python3 /path/to/your_script.py arg1 arg2";
    std::string output = execCommand(pythonScript);

    // 出力の表示
    std::cout << "Pythonの出力:" << std::endl;
    std::cout << output << std::endl;

    // リソースの解放
    delete[] inputData;
    delete[] outputPolicyData;
    delete[] outputValueData;
    cudaFree(buffers[inputIndex]);
    cudaFree(buffers[outputPolicyIndex]);
    cudaFree(buffers[outputValueIndex]);
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}