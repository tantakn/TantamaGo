/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/




/* 下ので動いた。buildディレクトリは作る必要なかったかも。
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/.local/mytensorrt/TensorRT-10.7.0.23/samples/sampleOnnxMNIST$ cd build/
cmake ..
make
cd ..
make
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/.local/mytensorrt/TensorRT-10.7.0.23/samples/sampleOnnxMNIST$ ../../bin/sample_onnx_mnist　mnistならこっち
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/.local/mytensorrt/TensorRT-10.7.0.23/samples/sampleOnnxMNIST$ ../../bin/sample_onnx_igo
*/


/*bin, data, include, lib, samples, をコピーして、makeとかしたらそこでも動いた。

make VERBOSE=1 でCMakeの出力を見れるらしい。多分、一時ファイルとか作ってるから g++ コマンドがいっぱいある。
tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/tensorRT_test/mini/samples/sampleOnnxMNIST$ make VERBOSE=1
../Makefile.config:25: CUDA_INSTALL_DIR variable is not specified, using /usr/local/cuda by default, use CUDA_INSTALL_DIR=<cuda_directory> to change.
../Makefile.config:45: TRT_LIB_DIR is not specified, searching ../../lib, ../../lib, ../lib by default, use TRT_LIB_DIR=<trt_lib_directory> to change.
if [ ! -d ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST ]; then mkdir -p ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST; fi
if [ ! -d ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common ]; then mkdir -p ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common; fi &&  if [ ! -d ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils ]; then mkdir -p ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils; fi && :
g++ -MM -MF ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.d -MP -MT ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o -Wall -Wno-deprecated-declarations -std=c++17 -I"../common" -I"../utils" -I".." -I"/usr/local/cuda/include" -I"../include" -I"../../include" -I"../../parsers/onnxOpenSource" -D_REENTRANT -DTRT_STATIC=0 igo.cpp
Compiling: igo.cpp
g++ -Wall -Wno-deprecated-declarations -std=c++17 -I"../common" -I"../utils" -I".." -I"/usr/local/cuda/include" -I"../include" -I"../../include" -I"../../parsers/onnxOpenSource" -D_REENTRANT -DTRT_STATIC=0 -g -c -o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o igo.cpp
Linking: ../../bin/sample_onnx_mnist_debug
g++ -o ../../bin/sample_onnx_mnist_debug -L"/usr/local/cuda/lib64" -Wl,-rpath-link="/usr/local/cuda/lib64" -L"../lib" -L"../../lib" -L"../../lib" -Wl,-rpath-link="../../lib"  -L"" -Wl,-rpath-link="" -L../../bin  -Wl,--start-group -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart -lrt -ldl -lpthread  ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/bfloat16.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/getOptions.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/logger.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleDevice.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleEngines.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleInference.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleOptions.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleReporting.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleUtils.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils/fileLock.o ../../bin/dchobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils/timingCache.o -Wl,--end-group -Wl,--no-relax
if [ ! -d ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST ]; then mkdir -p ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST; fi
if [ ! -d ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common ]; then mkdir -p ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common; fi &&  if [ ! -d ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils ]; then mkdir -p ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils; fi && :
g++ -MM -MF ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.d -MP -MT ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o -Wall -Wno-deprecated-declarations -std=c++17 -I"../common" -I"../utils" -I".." -I"/usr/local/cuda/include" -I"../include" -I"../../include" -I"../../parsers/onnxOpenSource" -D_REENTRANT -DTRT_STATIC=0 igo.cpp
Compiling: igo.cpp
g++ -Wall -Wno-deprecated-declarations -std=c++17 -I"../common" -I"../utils" -I".." -I"/usr/local/cuda/include" -I"../include" -I"../../include" -I"../../parsers/onnxOpenSource" -D_REENTRANT -DTRT_STATIC=0 -c -o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o igo.cpp
Linking: ../../bin/sample_onnx_igo
g++ -o ../../bin/sample_onnx_igo -L"/usr/local/cuda/lib64" -Wl,-rpath-link="/usr/local/cuda/lib64" -L"../lib" -L"../../lib" -L"../../lib" -Wl,-rpath-link="../../lib"  -L"" -Wl,-rpath-link="" -L../../bin  -Wl,--start-group -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart -lrt -ldl -lpthread  ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/igo.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/bfloat16.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/getOptions.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/logger.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleDevice.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleEngines.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleInference.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleOptions.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleReporting.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../common/sampleUtils.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils/fileLock.o ../../bin/chobj/sampleOnnxMNIST/sampleOnnxMNIST/../utils/timingCache.o -Wl,--end-group -Wl,--no-relax

上のを一つにまとめるとこうなるらしい。実行ディレクトリに sample_onnx_mnist_debug が出来て動いた。
tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/tensorRT_test/mini/samples/sampleOnnxMNIST$ g++ -Wall -Wno-deprecated-declarations -std=c++17   -I"../common"   -I"../utils"   -I".."   -I"/usr/local/cuda/include"   -I"../include"   -I"../../include"   -I"../../parsers/onnxOpenSource"   -D_REENTRANT -DTRT_STATIC=0   -g   igo.cpp   ../common/bfloat16.cpp   ../common/getOptions.cpp   ../common/logger.cpp   ../common/sampleDevice.cpp   ../common/sampleEngines.cpp   ../common/sampleInference.cpp   ../common/sampleOptions.cpp   ../common/sampleReporting.cpp   ../common/sampleUtils.cpp   ../utils/fileLock.cpp   ../utils/timingCache.cpp   -o sample_onnx_mnist_debug   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/lib64"   -L"../lib"   -L"../../lib"   -Wl,-rpath-link="../../lib"   -L"../../bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax
*/

/* ヘッダファイルとかのフォルダをコピペしても動いた
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/tensorRT_test/tiny$ tree -a -L 2
.
├── TensorRT
│   ├── bin
│   ├── common
│   ├── data
│   ├── include
│   ├── lib
│   └── utils
├── igo.cpp
├── json.hpp
└── test2.onnx

8 directories, 3 files
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/tensorRT_test/tiny$ g++ -Wall -Wno-deprecated-declarations -std=c++17   -I"./TensorRT/common"   -I"./TensorRT/utils"   -I"./TensorRT"   -I"/usr/local/cuda/include"   -I"./TensorRT/include"   -D_REENTRANT -DTRT_STATIC=0   -g   igo.cpp   ./TensorRT/common/bfloat16.cpp   ./TensorRT/common/getOptions.cpp   ./TensorRT/common/logger.cpp   ./TensorRT/common/sampleDevice.cpp   ./TensorRT/common/sampleEngines.cpp   ./TensorRT/common/sampleInference.cpp   ./TensorRT/common/sampleOptions.cpp   ./TensorRT/common/sampleReporting.cpp   ./TensorRT/common/sampleUtils.cpp   ./TensorRT/utils/fileLock.cpp   ./TensorRT/utils/timingCache.cpp   -o tensorIgo   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/lib64"   -L"./TensorRT/lib"   -Wl,-rpath-link="./TensorRT/lib"   -L"./TensorRT/bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax
*/

/*
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ g++ -Wall -Wno-deprecated-declarations -std=c++17   -I"./TensorRT/common"   -I"./TensorRT/utils"   -I"./TensorRT"   -I"/usr/local/cuda/include"   -I"./TensorRT/include"   -D_REENTRANT -DTRT_STATIC=0   -g   tensorRTigo.cpp   ./TensorRT/common/bfloat16.cpp   ./TensorRT/common/getOptions.cpp   ./TensorRT/common/logger.cpp   ./TensorRT/common/sampleDevice.cpp   ./TensorRT/common/sampleEngines.cpp   ./TensorRT/common/sampleInference.cpp   ./TensorRT/common/sampleOptions.cpp   ./TensorRT/common/sampleReporting.cpp   ./TensorRT/common/sampleUtils.cpp   ./TensorRT/utils/fileLock.cpp   ./TensorRT/utils/timingCache.cpp   -o tensorRTIgo   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/lib64"   -L"./TensorRT/lib"   -Wl,-rpath-link="./TensorRT/lib"   -L"./TensorRT/bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax
*/



//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!



// Define TRT entrypoints used in common code
#define DEFINE_TRT_ENTRYPOINTS 1
#define DEFINE_TRT_LEGACY_PARSER_ENTRYPOINT 0

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class TensorRTOnnxIgo
{
public:
    TensorRTOnnxIgo(const samplesCommon::OnnxSampleParams& params)
        : mParams(params)
        , mRuntime(nullptr)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer(const std::vector<std::vector<std::vector<float>>> inputPlane, std::vector<float> &outputPolicy, std::vector<float> &outputValue);

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;        //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputPolicyDims; //!< The dimensions of the output to the network.
    nvinfer1::Dims mOutputValueDims;  //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, const std::vector<std::vector<std::vector<float>>>);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool TensorRTOnnxIgo::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto timingCache = SampleUniquePtr<nvinfer1::ITimingCache>();

    auto constructed = constructNetwork(builder, network, config, parser, timingCache);
    if (!constructed)
    {
        return false;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    if (timingCache != nullptr && !mParams.timingCacheFile.empty())
    {
        samplesCommon::updateTimingCacheFile(
            sample::gLogger.getTRTLogger(), mParams.timingCacheFile, timingCache.get(), *builder);
    }

    auto profile = builder->createOptimizationProfile();////
    profile->setDimensions("input", OptProfileSelector::kMIN, nvinfer1::Dims4(1, 6, BOARDSIZE, BOARDSIZE));
    profile->setDimensions("input", OptProfileSelector::kOPT, nvinfer1::Dims4(1, 6, BOARDSIZE, BOARDSIZE));
    profile->setDimensions("input", OptProfileSelector::kMAX, nvinfer1::Dims4(1, 6, BOARDSIZE, BOARDSIZE));
    config->addOptimizationProfile(profile);

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 2);
    mOutputPolicyDims = network->getOutput(0)->getDimensions();
    assert(mOutputPolicyDims.nbDims == 2);
    mOutputValueDims = network->getOutput(1)->getDimensions();
    assert(mOutputValueDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool TensorRTOnnxIgo::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser, SampleUniquePtr<nvinfer1::ITimingCache>& timingCache)
{
    auto parsed = parser->parseFromFile(locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    if (mParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.bf16)
    {
        config->setFlag(BuilderFlag::kBF16);
    }
    if (mParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    if (mParams.timingCacheFile.size())
    {
        timingCache = samplesCommon::buildTimingCacheFromFile(
            sample::gLogger.getTRTLogger(), *config, mParams.timingCacheFile, sample::gLogError);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool TensorRTOnnxIgo::infer(const std::vector<std::vector<std::vector<float>>> inputPlane, std::vector<float> &outputPolicy, std::vector<float> &outputValue)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    for (int32_t i = 0, e = mEngine->getNbIOTensors(); i < e; i++)
    {
        auto const name = mEngine->getIOTensorName(i);
        context->setTensorAddress(name, buffers.getDeviceBuffer(name));
    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers, inputPlane))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }


    const int outputSize = mOutputPolicyDims.d[1];
    std::string outputPName = "output_policy";
    float *tmpOutputPolicy = static_cast<float *>(buffers.getHostBuffer(outputPName));
    const int outputValueSize = mOutputValueDims.d[1];
    std::string outputVName = "output_value";
    float *tmpOutputValue = static_cast<float *>(buffers.getHostBuffer(outputVName));

    for (int i = 0; i < outputSize; i++)
    {
        // cerr << "outputPolicy[" << i << "] = " << tmpOutputPolicy[i] << std::endl;
        outputPolicy[i] = tmpOutputPolicy[i];
    }
    for (int i = 0; i < outputValueSize; i++)
    {
        // cerr << "outputValue[" << i << "] = " << tmpOutputValue[i] << std::endl;
        outputValue.push_back(tmpOutputValue[i]);
    }

    return true;
}


std::string addProfileSuffix(const std::string &name, int profile)
{
    std::ostringstream oss;
    oss << name;
    if (profile > 0)
    {
        oss << " [profile " << profile << "]";
    }

    return oss.str();
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool TensorRTOnnxIgo::processInput(const samplesCommon::BufferManager& buffers, const std::vector<std::vector<std::vector<float>>> inputPlane)
{
    const int inputC = mInputDims.d[1];
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::string inputName = "input";
    // std::string inputName = addProfileSuffix(inputTensorNames[0], profileForBatchSize[batchSize]);

    float *hostDataBuffer = static_cast<float *>(buffers.getHostBuffer(inputName));

    memset(hostDataBuffer, 0, inputC * inputH * inputW * sizeof(float)); // メモリの確保
    // memset(hostDataBuffer, 0, inputC * inputH * inputW * sizeof(float) * batchSize); // メモリの確保

    // nlohmann::json js = nlohmann::json::parse(inputJson);


    // // 確認用の出力。
    // sample::gLogInfo << std::endl;
    // for (int i = 0; i < inputH; i++) {
    //     for (int j = 0; j < inputW; j++) {
    //         if (js[0][i][j] == 1.0) {
    //             sample::gLogInfo << " _";
    //         }
    //         else if (js[1][i][j] == 1.0) {
    //             sample::gLogInfo << " #";
    //         }
    //         else {
    //             sample::gLogInfo << " O";
    //         }
    //     }
    //     sample::gLogInfo << std::endl;
    // }


    for (int i = 0; i < inputC; i++)
    {
        for (int j = 0; j < inputH; j++)
        {
            for (int k = 0; k < inputW; k++)
            {
                hostDataBuffer[i * inputH * inputW + j * inputW + k] = inputPlane[i][j][k];
            }
        }
    }


    return true;

    // const int inputH = mInputDims.d[2];
    // const int inputW = mInputDims.d[3];

    // // Read a random digit file
    // srand(unsigned(time(nullptr)));
    // std::vector<uint8_t> fileData(inputH * inputW);
    // mNumber = rand() % 10;
    // readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // // Print an ascii representation
    // sample::gLogInfo << "Input:" << std::endl;
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    // }
    // sample::gLogInfo << std::endl;

    // float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    // for (int i = 0; i < inputH * inputW; i++)
    // {
    //     hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    // }

    // return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool TensorRTOnnxIgo::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputPolicyDims.d[1];
    std::string outputPName = "output_policy";
    float *outputPolicy = static_cast<float *>(buffers.getHostBuffer(outputPName));
    const int outputValueSize = mOutputValueDims.d[1];
    std::string outputVName = "output_value";
    float *outputValue = static_cast<float *>(buffers.getHostBuffer(outputVName));
    
    // for (int i = 0; i < outputSize; i++)
    // {
    //     sample::gLogInfo << "outputPolicy[" << i << "] = " << outputPolicy[i] << std::endl;
    // }
    // for (int i = 0; i < outputValueSize; i++)
    // {
    //     sample::gLogInfo << "outputValue[" << i << "] = " << outputValue[i] << std::endl;
    // }
    return true;

    // return true;

    // // const int outputSize = mOutputDims.d[1];
    // // float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    // // float val{0.0F};
    // // int idx{0};

    // // // Calculate Softmax
    // // float sum{0.0F};
    // // for (int i = 0; i < outputSize; i++)
    // // {
    // //     output[i] = exp(output[i]);
    // //     sum += output[i];
    // // }

    // // sample::gLogInfo << "Output:" << std::endl;
    // // for (int i = 0; i < outputSize; i++)
    // // {
    // //     output[i] /= sum;
    // //     val = std::max(val, output[i]);
    // //     if (val == output[i])
    // //     {
    // //         idx = i;
    // //     }

    // //     sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
    // //                      << " "
    // //                      << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
    // //                      << std::endl;
    // // }
    // // sample::gLogInfo << std::endl;

    // // return idx == mNumber && val > 0.9F;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args, std::string model)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    { // ここにモデルを置く。
        // params.dataDirs.push_back("data/mnist/");
        // params.dataDirs.push_back("data/samples/mnist/");
        params.dataDirs.push_back(".");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = model; // ここでモデルを指定する。
    // params.onnxFileName = "test19_2.onnx"; // ここでモデルを指定する。
    // params.onnxFileName = "test2.onnx"; // ここでモデルを指定する。
    // params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;
    // params.int8 = false;
    // params.fp16 = true;
    // params.bf16 = true;
    // // params.int8 = args.runInInt8;
    // // params.fp16 = args.runInFp16;
    // // params.bf16 = args.runInBf16;
    params.timingCacheFile = args.timingCacheFile;

    return params;
}
// //!
// //! \brief Initializes members of the params struct using the command line args
// //!
// samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
// {
//     samplesCommon::OnnxSampleParams params;
//     if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
//     { // ここにモデルを置く。
//         params.dataDirs.push_back("data/mnist/");
//         params.dataDirs.push_back("data/samples/mnist/");
//         params.dataDirs.push_back(".");
//     }
//     else // Use the data directory provided by the user
//     {
//         params.dataDirs = args.dataDirs;
//     }
//     params.onnxFileName = "test2.onnx"; // ここでモデルを指定する。
//     // params.onnxFileName = "mnist.onnx";
//     params.inputTensorNames.push_back("Input3");
//     params.outputTensorNames.push_back("Plus214_Output_0");
//     params.dlaCore = args.useDLACore;
//     params.int8 = false;
//     params.fp16 = false;
//     params.bf16 = false;
//     // params.int8 = false;
//     // params.fp16 = true;
//     // params.bf16 = true;
//     // // params.int8 = args.runInInt8;
//     // // params.fp16 = args.runInFp16;
//     // // params.bf16 = args.runInBf16;
//     params.timingCacheFile = args.timingCacheFile;

//     return params;
// }

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << "[-t or --timingCacheFile=<path to timing cache file]" << std::endl;
    std::cout << "--help             Display help information" << std::endl;
    std::cout << "--datadir          Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N     Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8             Run in Int8 mode." << std::endl;
    std::cout << "--fp16             Run in FP16 mode." << std::endl;
    std::cout << "--bf16             Run in BF16 mode." << std::endl;
    std::cout << "--timingCacheFile  Specify path to a timing cache file. If it does not already exist, it will be "
              << "created." << std::endl;
}

// int main(int argc, char** argv)
// {
//     std::string inputStr = "[[[0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],[1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],[[1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0],[1.0,0.0,1.0,1.0,1.0,0.0,1.0,0.0,1.0],[0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0]],[[0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0],[1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0],[0.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0],[0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0],[1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]]";
//     nlohmann::json inputJson = nlohmann::json::parse(inputStr);
//     std::vector<std::vector<std::vector<float>>> inputPlane = inputJson;


//     samplesCommon::Args args;

//     args.runInInt8 = false;
//     args.runInFp16 = false;
//     args.runInBf16 = false;

//     TensorRTOnnxIgo sample(initializeSampleParams(args, "test2.onnx"));

//     sample.build();

//     std::vector<float> outputPolicy;
//     std::vector<float> outputValue;


//     sample.infer(inputPlane, outputPolicy, outputValue);
//     print(inputPlane);
//     print(outputPolicy);
//     print(outputValue);

//     return 0;
//     // samplesCommon::Args args;
//     // bool argsOK = samplesCommon::parseArgs(args, argc, argv);
//     // if (!argsOK)
//     // {
//     //     sample::gLogError << "Invalid arguments" << std::endl;
//     //     printHelpInfo();
//     //     return EXIT_FAILURE;
//     // }
//     // if (args.help)
//     // {
//     //     printHelpInfo();
//     //     return EXIT_SUCCESS;
//     // }

//     // args.runInInt8 = false;
//     // args.runInFp16 = false;
//     // args.runInBf16 = false;

//     // auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

//     // sample::gLogger.reportTestStart(sampleTest);

//     // TensorRTOnnxIgo sample(initializeSampleParams(args, "test2.onnx"));

//     // if (!sample.build())
//     // {
//     //     return sample::gLogger.reportFail(sampleTest);
//     // }

//     // if (!sample.infer(inputJson))
//     // {
//     //     return sample::gLogger.reportFail(sampleTest);
//     // }

//     // return sample::gLogger.reportPass(sampleTest);
// }



// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/tensorRT_test/mini/samples/sampleOnnxMNIST$ ./sample_onnx_mnist_debug 
// &&&& RUNNING TensorRT.sample_onnx_mnist [TensorRT v100700] [b23] # ./sample_onnx_mnist_debug
// [01/12/2025-02:08:40] [I] Building a GPU inference engine for Onnx MNIST
// [01/12/2025-02:08:41] [I] [TRT] [MemUsageChange] Init CUDA: CPU +20, GPU +0, now: CPU 22, GPU 1040 (MiB)
// [01/12/2025-02:08:47] [I] [TRT] [MemUsageChange] Init builder kernel library: CPU +2175, GPU +406, now: CPU 2352, GPU 1446 (MiB)
// [01/12/2025-02:08:47] [I] [TRT] ----------------------------------------------------------------
// [01/12/2025-02:08:47] [I] [TRT] Input filename:   ../../data/mnist/test2.onnx
// [01/12/2025-02:08:47] [I] [TRT] ONNX IR version:  0.0.5
// [01/12/2025-02:08:47] [I] [TRT] Opset version:    10
// [01/12/2025-02:08:47] [I] [TRT] Producer name:    pytorch
// [01/12/2025-02:08:47] [I] [TRT] Producer version: 2.5.0
// [01/12/2025-02:08:47] [I] [TRT] Domain:           
// [01/12/2025-02:08:47] [I] [TRT] Model version:    0
// [01/12/2025-02:08:47] [I] [TRT] Doc string:       
// [01/12/2025-02:08:47] [I] [TRT] ----------------------------------------------------------------
// [01/12/2025-02:08:47] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
// [01/12/2025-02:08:49] [I] [TRT] Compiler backend is used during engine build.
// [01/12/2025-02:08:50] [I] [TRT] Detected 1 inputs and 2 output network tensors.
// [01/12/2025-02:08:50] [I] [TRT] Total Host Persistent Memory: 86816 bytes
// [01/12/2025-02:08:50] [I] [TRT] Total Device Persistent Memory: 0 bytes
// [01/12/2025-02:08:50] [I] [TRT] Max Scratch Memory: 0 bytes
// [01/12/2025-02:08:50] [I] [TRT] [BlockAssignment] Started assigning block shifts. This will take 19 steps to complete.
// [01/12/2025-02:08:50] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.049396ms to assign 3 blocks to 19 nodes requiring 62976 bytes.
// [01/12/2025-02:08:50] [I] [TRT] Total Activation Memory: 62976 bytes
// [01/12/2025-02:08:50] [I] [TRT] Total Weights Memory: 1851804 bytes
// [01/12/2025-02:08:50] [I] [TRT] Compiler backend is used during engine execution.
// [01/12/2025-02:08:50] [I] [TRT] Engine generation completed in 3.44483 seconds.
// [01/12/2025-02:08:50] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 0 MiB, GPU 6 MiB
// [01/12/2025-02:08:50] [I] [TRT] Loaded engine size: 2 MiB
// [01/12/2025-02:08:50] [I] running a GPU inference engine for Onnx MNIST
// [01/12/2025-02:08:50] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 1 (MiB)
// [01/12/2025-02:08:50] [I] 
// [01/12/2025-02:08:50] [I]  # # _ O O _ O O O
// [01/12/2025-02:08:50] [I]  O O O O # _ O O O
// [01/12/2025-02:08:50] [I]  O O O _ _ O O O #
// [01/12/2025-02:08:50] [I]  _ _ _ _ _ O O # #
// [01/12/2025-02:08:50] [I]  _ O _ O O O # _ #
// [01/12/2025-02:08:50] [I]  _ O O _ O O # # _
// [01/12/2025-02:08:50] [I]  O _ O O O # # # #
// [01/12/2025-02:08:50] [I]  # O # # # _ # _ #
// [01/12/2025-02:08:50] [I]  _ O # # # # _ # #
// [01/12/2025-02:08:50] [I] outputPolicy[0] = -1.11348
// [01/12/2025-02:08:50] [I] outputPolicy[1] = -0.804844
// [01/12/2025-02:08:50] [I] outputPolicy[2] = 2.25714
// [01/12/2025-02:08:50] [I] outputPolicy[3] = 0.148132
// [01/12/2025-02:08:50] [I] outputPolicy[4] = 0.61172
// [01/12/2025-02:08:50] [I] outputPolicy[5] = 6.09207
// [01/12/2025-02:08:50] [I] outputPolicy[6] = -0.457173
// [01/12/2025-02:08:50] [I] outputPolicy[7] = -1.47292
// [01/12/2025-02:08:50] [I] outputPolicy[8] = -0.859812
// [01/12/2025-02:08:50] [I] outputPolicy[9] = -1.14788
// [01/12/2025-02:08:50] [I] outputPolicy[10] = -2.16358
// [01/12/2025-02:08:50] [I] outputPolicy[11] = -1.39354
// [01/12/2025-02:08:50] [I] outputPolicy[12] = -0.858219
// [01/12/2025-02:08:50] [I] outputPolicy[13] = 0.273903
// [01/12/2025-02:08:50] [I] outputPolicy[14] = 5.73078
// [01/12/2025-02:08:50] [I] outputPolicy[15] = -0.7942
// [01/12/2025-02:08:50] [I] outputPolicy[16] = -2.74528
// [01/12/2025-02:08:50] [I] outputPolicy[17] = -1.32528
// [01/12/2025-02:08:50] [I] outputPolicy[18] = -1.42789
// [01/12/2025-02:08:50] [I] outputPolicy[19] = -1.33704
// [01/12/2025-02:08:50] [I] outputPolicy[20] = -0.392883
// [01/12/2025-02:08:50] [I] outputPolicy[21] = 3.19206
// [01/12/2025-02:08:50] [I] outputPolicy[22] = 6.04731
// [01/12/2025-02:08:50] [I] outputPolicy[23] = -0.414737
// [01/12/2025-02:08:50] [I] outputPolicy[24] = -0.803619
// [01/12/2025-02:08:50] [I] outputPolicy[25] = -2.78288
// [01/12/2025-02:08:50] [I] outputPolicy[26] = -1.41846
// [01/12/2025-02:08:50] [I] outputPolicy[27] = 3.6092
// [01/12/2025-02:08:50] [I] outputPolicy[28] = 3.45132
// [01/12/2025-02:08:50] [I] outputPolicy[29] = 5.14762
// [01/12/2025-02:08:50] [I] outputPolicy[30] = 3.99032
// [01/12/2025-02:08:50] [I] outputPolicy[31] = 1.75471
// [01/12/2025-02:08:50] [I] outputPolicy[32] = -1.90241
// [01/12/2025-02:08:50] [I] outputPolicy[33] = -1.55602
// [01/12/2025-02:08:50] [I] outputPolicy[34] = -1.64266
// [01/12/2025-02:08:50] [I] outputPolicy[35] = -1.51641
// [01/12/2025-02:08:50] [I] outputPolicy[36] = 3.91962
// [01/12/2025-02:08:50] [I] outputPolicy[37] = -0.728382
// [01/12/2025-02:08:50] [I] outputPolicy[38] = 3.22004
// [01/12/2025-02:08:50] [I] outputPolicy[39] = -0.960262
// [01/12/2025-02:08:50] [I] outputPolicy[40] = -2.18892
// [01/12/2025-02:08:50] [I] outputPolicy[41] = -1.7764
// [01/12/2025-02:08:50] [I] outputPolicy[42] = -2.68838
// [01/12/2025-02:08:50] [I] outputPolicy[43] = 2.14384
// [01/12/2025-02:08:50] [I] outputPolicy[44] = -1.03382
// [01/12/2025-02:08:50] [I] outputPolicy[45] = 5.81144
// [01/12/2025-02:08:50] [I] outputPolicy[46] = -1.19603
// [01/12/2025-02:08:50] [I] outputPolicy[47] = -1.99508
// [01/12/2025-02:08:50] [I] outputPolicy[48] = -1.23695
// [01/12/2025-02:08:50] [I] outputPolicy[49] = -2.28582
// [01/12/2025-02:08:50] [I] outputPolicy[50] = -2.70703
// [01/12/2025-02:08:50] [I] outputPolicy[51] = -1.97409
// [01/12/2025-02:08:50] [I] outputPolicy[52] = -2.39571
// [01/12/2025-02:08:50] [I] outputPolicy[53] = 3.56375
// [01/12/2025-02:08:50] [I] outputPolicy[54] = 0.529106
// [01/12/2025-02:08:50] [I] outputPolicy[55] = 2.24873
// [01/12/2025-02:08:50] [I] outputPolicy[56] = -3.03173
// [01/12/2025-02:08:50] [I] outputPolicy[57] = -2.43669
// [01/12/2025-02:08:50] [I] outputPolicy[58] = -2.7211
// [01/12/2025-02:08:50] [I] outputPolicy[59] = -2.92571
// [01/12/2025-02:08:50] [I] outputPolicy[60] = -2.90276
// [01/12/2025-02:08:50] [I] outputPolicy[61] = -1.87872
// [01/12/2025-02:08:50] [I] outputPolicy[62] = 0.174346
// [01/12/2025-02:08:50] [I] outputPolicy[63] = -0.0478945
// [01/12/2025-02:08:50] [I] outputPolicy[64] = -0.659336
// [01/12/2025-02:08:50] [I] outputPolicy[65] = -2.31873
// [01/12/2025-02:08:50] [I] outputPolicy[66] = -1.68059
// [01/12/2025-02:08:50] [I] outputPolicy[67] = -2.54881
// [01/12/2025-02:08:50] [I] outputPolicy[68] = 3.2844
// [01/12/2025-02:08:50] [I] outputPolicy[69] = -2.07118
// [01/12/2025-02:08:50] [I] outputPolicy[70] = 4.09083
// [01/12/2025-02:08:50] [I] outputPolicy[71] = -0.456248
// [01/12/2025-02:08:50] [I] outputPolicy[72] = 2.67175
// [01/12/2025-02:08:50] [I] outputPolicy[73] = -1.64613
// [01/12/2025-02:08:50] [I] outputPolicy[74] = -0.848888
// [01/12/2025-02:08:50] [I] outputPolicy[75] = -1.61496
// [01/12/2025-02:08:50] [I] outputPolicy[76] = -1.05942
// [01/12/2025-02:08:50] [I] outputPolicy[77] = -0.595472
// [01/12/2025-02:08:50] [I] outputPolicy[78] = 4.49476
// [01/12/2025-02:08:50] [I] outputPolicy[79] = 0.00886209
// [01/12/2025-02:08:50] [I] outputPolicy[80] = 0.492329
// [01/12/2025-02:08:50] [I] outputPolicy[81] = 7.45327
// [01/12/2025-02:08:50] [I] outputValue[0] = 5.37408
// [01/12/2025-02:08:50] [I] outputValue[1] = -6.50008
// [01/12/2025-02:08:50] [I] outputValue[2] = 1.08489
// [01/12/2025-02:08:50] [I] succeed2
// &&&& PASSED TensorRT.sample_onnx_mnist [TensorRT v100700] [b23] # ./sample_onnx_mnist_debug