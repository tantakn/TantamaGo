/**
 * @file include/dl/trt/TensorRTNetwork.hpp
 * @author Yuki Kobayashi
 * @~english
 * @brief Neural network using TensorRT.
 * @~japanese
 * @brief TensorRTを使ったニューラルネットワークの実装
 */
#ifndef _TENSOR_RT_NETWORK_HPP_
#define _TENSOR_RT_NETWORK_HPP_

#define USE_TENSOR_RT  ////////

#if defined USE_TENSOR_RT

#include <cassert>
#include <fstream>
#include <mutex>

#include "NvInfer.h"
#include "NvInferRuntimeCommon.h"
#include "NvOnnxParser.h"
// #include "dl/CudaWrapper.hpp"
// #include "dl/RayNet.hpp"
// #include "dl/cuda/ConvertInputData.hpp"
// #include "dl/trt/TensorRTLogger.hpp"
// #include "feature/FeatureUtility.hpp"
// #include "feature/NeuralFeature.hpp"


constexpr long long int operator"" _MiB(const unsigned long long size)
{
    return size * (1 << 20);
}


struct InferDeleter {
    template <typename T>
    void operator()(T *obj) const
    {
        if (obj) {
#if NV_TENSORRT_MAJOR >= 8
            delete obj;
#else
            obj->destroy();
#endif
        }
    }
};


template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;


class NNTensorRT : public RayNet {
   private:
    static constexpr int features = NN_FEATURES;
    const int gpu_id;
    const int max_batch_size;
    const int plane_size;
    const int policy_size;
    InferUniquePtr<nvinfer1::ICudaEngine> engine;
    TensorRTLogger gLogger;

    unsigned long long *bit_ptr;
    cuda_float_t *real_ptr;

    cuda_float_t *input_ptr;
    cuda_float_t *policy_ptr;
    cuda_float_t *value_ptr;
    cuda_float_t *score_ptr;
    cuda_float_t *owner_ptr;
#if defined USE_FP16
    float *f_policy_ptr;
    float *f_value_ptr;
    float *f_score_ptr;
    float *f_owner_ptr;
#endif
    std::vector<void *> input_bindings;
    InferUniquePtr<nvinfer1::IExecutionContext> context;
    nvinfer1::Dims input_dims;
    std::mutex mutex;

   public:
    NNTensorRT(const char *filename, const int gpu_id, const int max_batch_size)
        : gpu_id(gpu_id), max_batch_size(max_batch_size), plane_size(pure_board_size), policy_size(pure_board_max + 1)
    {
        CheckCudaErrors(__FILE__, __LINE__, cudaSetDevice(gpu_id));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&bit_ptr, sizeof(unsigned long long) * max_batch_size * plane_size * plane_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&real_ptr, sizeof(cuda_float_t) * max_batch_size * NN_REAL_NUMBER_MAX));

        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&input_ptr, sizeof(cuda_float_t) * max_batch_size * NN_FEATURES * plane_size * plane_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&policy_ptr, sizeof(cuda_float_t) * max_batch_size * policy_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&value_ptr, sizeof(cuda_float_t) * max_batch_size * 3));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&score_ptr, sizeof(cuda_float_t) * max_batch_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&owner_ptr, sizeof(cuda_float_t) * max_batch_size * plane_size * plane_size));
#if defined USE_FP16
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&f_policy_ptr, sizeof(float) * max_batch_size * policy_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&f_value_ptr, sizeof(float) * max_batch_size * 3));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&f_score_ptr, sizeof(float) * max_batch_size));
        CheckCudaErrors(__FILE__, __LINE__, cudaMalloc((void **)&f_owner_ptr, sizeof(float) * max_batch_size * plane_size * plane_size));
#endif

        input_bindings = {input_ptr, policy_ptr, score_ptr, value_ptr, owner_ptr};

        load_model(filename);
    }

    ~NNTensorRT(void)
    {
        CheckCudaErrors(__FILE__, __LINE__, cudaSetDevice(gpu_id));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(bit_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(real_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(input_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(policy_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(value_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(score_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(owner_ptr));

#if defined USE_FP16
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(f_policy_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(f_value_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(f_score_ptr));
        CheckCudaErrors(__FILE__, __LINE__, cudaFree(f_owner_ptr));
#endif
    }


    void
    build_model(const std::string &onnx_filename)
    {
        InferUniquePtr<nvinfer1::IBuilder> builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));

        if (!builder) {
            throw std::runtime_error("createInferBuilder");
        }

        const nvinfer1::NetworkDefinitionCreationFlags explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        // const auto explicit_batch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        InferUniquePtr<nvinfer1::INetworkDefinition> network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicit_batch));
        if (!network) {
            throw std::runtime_error("createNetworkV2");
        }

        InferUniquePtr<nvinfer1::IBuilderConfig> config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("createBilderConfig");
        }

        InferUniquePtr<nvonnxparser::IParser> parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            throw std::runtime_error("createParser");
        }

        const bool parsed = parser->parseFromFile(onnx_filename.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));
        if (!parsed) {
            throw std::runtime_error("parseFromFile");
        }

        builder->setMaxBatchSize(max_batch_size);
        config->setMaxWorkspaceSize(512_MiB);

        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

#if defined USE_FP16
        network->getInput(0)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(0)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(1)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(2)->setType(nvinfer1::DataType::kHALF);
        network->getOutput(3)->setType(nvinfer1::DataType::kHALF);
#endif

        for (int i = 0; i < 4; i++) {
            nvinfer1::Dims outputDims = network->getOutput(i)->getDimensions();
            const int32_t *dims = outputDims.d;
            std::cerr << "Output " << i << " : " << dims[0] << "," << dims[1] << "," << dims[2] << "," << dims[3] << std::endl;
        }

        assert(network->getNbInputs() == 1);
        nvinfer1::Dims input_dims = network->getInput(0)->getDimensions();
        assert(input_dims.nbDims == 4);
        assert(network->getNbOutputs() == 4);


        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        const int32_t *dims = input_dims.d;

        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims[1], dims[2], dims[3]));
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims[1], dims[2], dims[3]));
        profile->setDimensions("input", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims[1], dims[2], dims[3]));
        config->addOptimizationProfile(profile);

#if NV_TENSORRT_MAJOR >= 8
        InferUniquePtr<nvinfer1::IHostMemory> serialized_engine = InferUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!serialized_engine) {
            throw std::runtime_error("buildSerializedNetwork");
        }

        InferUniquePtr<nvinfer1::IRuntime> runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        engine.reset(runtime->deserializeCudaEngine(serialized_engine->data(), serialized_engine->size()));
        if (!engine) {
            throw std::runtime_error("deserializedCudaEngine");
        }
#else
        engine.reset(builder->buildEngineWithConfig(*network, *config));
        if (!engine) {
            throw std::runtime_error("buildEngineWithConfig");
        }
#endif
    }


    void
    load_model(const char *filename)
    {
        std::string serialized_filename = std::string(filename) + "." + std::to_string(gpu_id) + "." + std::to_string(max_batch_size)
#if defined USE_FP16
                                          + ".fp16"
#endif
                                          + ".serialized";

        std::ifstream serialized_file(serialized_filename, std::ios::binary);
        if (serialized_file.is_open()) {
            serialized_file.seekg(0, std::ios_base::end);

            const size_t model_size = serialized_file.tellg();
            serialized_file.seekg(0, std::ios_base::beg);

            std::unique_ptr<char[]> blob(new char[model_size]);
            serialized_file.read(blob.get(), model_size);

            InferUniquePtr<nvinfer1::IRuntime> runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
            engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(blob.get(), model_size));
        }
        else {
            build_model(filename);

            InferUniquePtr<nvinfer1::IHostMemory> serialized_engine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
            if (!serialized_engine) {
                throw std::runtime_error("Engine serialization failed");
            }

            std::ofstream engine_file(serialized_filename, std::ios::binary);
            if (!engine_file) {
                throw std::runtime_error("Cannot open engine file");
            }

            engine_file.write(static_cast<char *>(serialized_engine->data()), serialized_engine->size());
            if (engine_file.fail()) {
                throw std::runtime_error("Cannot open engile file");
            }
        }

        context = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

        if (!context) {
            throw std::runtime_error("createExecutionContext");
        }

        input_dims = engine->getBindingDimensions(0);
    }

    void
    Forward(const unsigned long long bit_data[], const float real_data[], float policy_out[], float value_out[], float score_out[], float owner_out[], const int batch_size)
    {
        mutex.lock();

        int current_device_id;
        CheckCudaErrors(__FILE__, __LINE__, cudaGetDevice(&current_device_id));
        if (current_device_id != gpu_id) {
            CheckCudaErrors(__FILE__, __LINE__, cudaSetDevice(gpu_id));
        }

        input_dims.d[0] = batch_size;
        context->setBindingDimensions(0, input_dims);

        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(bit_ptr, bit_data, sizeof(unsigned long long) * batch_size * plane_size * plane_size, cudaMemcpyHostToDevice, cudaStreamPerThread));

#if defined USE_FP16
        cuda_float_t real_h_data[batch_size * NN_REAL_NUMBER_MAX];
        for (int i = 0; i < batch_size * NN_REAL_NUMBER_MAX; i++) {
            real_h_data[i] = __float2half(real_data[i]);
        }
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(real_ptr, real_h_data, sizeof(cuda_float_t) * batch_size * NN_REAL_NUMBER_MAX, cudaMemcpyHostToDevice, cudaStreamPerThread));
#else
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(real_ptr, real_data, sizeof(float) * batch_size * NN_REAL_NUMBER_MAX, cudaMemcpyHostToDevice, cudaStreamPerThread));
#endif

        ConvertInputDataNCHW(bit_ptr, real_ptr, input_ptr, batch_size * features * plane_size * plane_size);

        const bool status = context->enqueue(batch_size, input_bindings.data(), cudaStreamPerThread, nullptr);
        assert(status);

#if defined USE_FP16
        CastHalfData(policy_ptr, f_policy_ptr, batch_size * policy_size);
        CastHalfData(value_ptr, f_value_ptr, batch_size * 3);
        CastHalfData(score_ptr, f_score_ptr, batch_size);
        CastHalfData(owner_ptr, f_owner_ptr, batch_size * plane_size * plane_size);

        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(policy_out, f_policy_ptr, sizeof(float) * batch_size * policy_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(value_out, f_value_ptr, sizeof(float) * batch_size * 3, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(score_out, f_score_ptr, sizeof(float) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(owner_out, f_owner_ptr, sizeof(float) * batch_size * plane_size * plane_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
#else
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(policy_out, policy_ptr, sizeof(float) * batch_size * policy_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(value_out, value_ptr, sizeof(float) * batch_size * 3, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(score_out, score_ptr, sizeof(float) * batch_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
        CheckCudaErrors(__FILE__, __LINE__, cudaMemcpyAsync(owner_out, owner_ptr, sizeof(float) * batch_size * plane_size * plane_size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
#endif
        CheckCudaErrors(__FILE__, __LINE__, cudaStreamSynchronize(cudaStreamPerThread));

        mutex.unlock();
    }
};


#endif

#endif


int main() {
    char 
    const char *filename = "test.onnx";
    const int gpu_id = 0;
    const int max_batch_size = 1;
    
    NNTensorRT nn(filename, gpu_id, max_batch_size);
    return 0;
}