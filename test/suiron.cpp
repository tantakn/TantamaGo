// /usr/include/cuda.h
// g++ suiron.cpp -o suiron     -I/usr/include     -L/usr/lib/x86_64-linux-gnu     -lnvinfer -lnvonnxparser -lcudart -I/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include
// g++ suiron.cpp -o suiron     -I/usr/include     -I/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include     -L/usr/lib/x86_64-linux-gnu     -L/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/lib     -lnvinfer -lnvonnxparser -lcudart

// ↓動いた
// (env) (base) u2424004@g14:~/igo/TantamaGo/test$ export LD_LIBRARY_PATH=/home0/y2024/u2424004/.local/mycudnn/cuda/lib64:$LD_LIBRARY_PATH

// (env) (base) u2424004@g14:~/igo/TantamaGo/test$ g++ suiron.cpp -o suiron     -I/usr/include     -I/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/include     -I/home0/y2024/u2424004/.local/mycudnn/cuda/include     -L/usr/lib/x86_64-linux-gnu     -L/home0/y2024/u2424004/.local/mycudnn/TensorRT-6.0.1.5/lib     -L/home0/y2024/u2424004/.local/mycudnn/cuda/lib64     -Wl,-rpath,/home0/y2024/u2424004/.local/mycudnn/cuda/lib64     -lnvinfer -lnvonnxparser -lcudart -lcudnn

// (env) (base) u2424004@g14:~/igo/TantamaGo/test$ ./suiron
// ----------------------------------------------------------------
// Input filename:   test.onnx
// ONNX IR version:  0.0.5
// Opset version:    10
// Producer name:    pytorch
// Producer version: 1.13.1
// Domain:           
// Model version:    0
// Doc string:       
// ----------------------------------------------------------------
// WARNING: ONNX model has a newer ir_version (0.0.5) than this parser was built against (0.0.3).
// While parsing node number 0 [Conv]:
// ERROR: ModelImporter.cpp:296 In function importModel:
// [5] Assertion failed: tensors.count(input_name)
// Network must have at least one output
// Segmentation fault (コアダンプ)


#include <iostream>
#include <fstream>
// #include <NvInfer.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>

#include <bits/stdc++.h>

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
        // ネットワークの作成時にフラグを設定
        INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        // INetworkDefinition* network = builder->createNetworkV2(0);
        // ONNXパーサーの作成
        nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
        // ONNXモデルの読み込み
        parser->parseFromFile("test.onnx", static_cast<int>(ILogger::Severity::kWARNING));
        // エンジンのビルド
        IBuilderConfig* config = builder->createBuilderConfig();

        // 変更前
        // config->setMaxWorkspaceSize(1 << 20);  // 1MB

        // 変更後
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

        engine = builder->buildEngineWithConfig(*network, *config);
        // リソースの解放
        // parser->destroy();
        // network->destroy();
        // builder->destroy();
        // config->destroy();
    }

    // Contextの作成
    IExecutionContext* context = engine->createExecutionContext();

    // 入力データの準備（バッチサイズ1、チャンネル6、サイズ9x9）
    int batchSize = 1;
    int inputIndex = engine->getTensorIndex("input");
    Dims inputDims = engine->getTensorShape(inputIndex);
    // int inputIndex = engine->getBindingIndex("input");
    // Dims inputDims = engine->getBindingDimensions(inputIndex);
    size_t inputSize = 1;
    for (int i = 0; i < inputDims.nbDims; ++i)
        inputSize *= inputDims.d[i];
    float* inputData = new float[inputSize];
    // 入力データを設定（例として1.0で初期化）
    for (size_t i = 0; i < inputSize; ++i)
        inputData[i] = 1.0f;

    // 出力データの準備
    int outputPolicyIndex = engine->getBindingIndex("output_policy");
    // int outputPolicyIndex = engine->getTensorIndex("output_policy");
    Dims outputPolicyDims = engine->getTensorShape(outputPolicyIndex);
    // int outputValueIndex = engine->getBindingIndex("output_value");
    // Dims outputPolicyDims = engine->getBindingDimensions(outputPolicyIndex);
    size_t outputPolicySize = 1;
    for (int i = 0; i < outputPolicyDims.nbDims; ++i)
        outputPolicySize *= outputPolicyDims.d[i];
    float* outputPolicyData = new float[outputPolicySize];

    Dims outputValueDims = engine->getTensorShape(outputValueIndex);
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
    // context->destroy();
    // engine->destroy();
    // runtime->destroy();

    return 0;
}