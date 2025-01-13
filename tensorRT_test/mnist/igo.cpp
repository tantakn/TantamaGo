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

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!



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



#pragma once

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

const std::string gSampleName = "TensorRT.sample_onnx_mnist";

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
public:
    SampleOnnxMNIST(const samplesCommon::OnnxSampleParams& params)
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
    bool infer();

private:
    samplesCommon::OnnxSampleParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

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
bool SampleOnnxMNIST::build()
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

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
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

    ASSERT(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    ASSERT(mOutputDims.nbDims == 2);

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
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
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

    samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
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
    if (!processInput(buffers))
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

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    // Read a random digit file
    srand(unsigned(time(nullptr)));
    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = rand() % 10;
    readPGMFile(locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    // Print an ascii representation
    sample::gLogInfo << "Input:" << std::endl;
    for (int i = 0; i < inputH * inputW; i++)
    {
        sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    sample::gLogInfo << std::endl;

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0F};
    int idx{0};

    // Calculate Softmax
    float sum{0.0F};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    sample::gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        sample::gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i]
                         << " "
                         << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5F)), '*')
                         << std::endl;
    }
    sample::gLogInfo << std::endl;

    return idx == mNumber && val > 0.9F;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::OnnxSampleParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::OnnxSampleParams params;
    if (args.dataDirs.empty()) // Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("data/mnist/");
        params.dataDirs.push_back("data/samples/mnist/");
    }
    else // Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = "mnist.onnx";
    params.inputTensorNames.push_back("Input3");
    params.outputTensorNames.push_back("Plus214_Output_0");
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;
    params.bf16 = args.runInBf16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]"
        << std::endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
    std::cout << "--bf16          Run in BF16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(sampleTest);

    SampleOnnxMNIST sample(initializeSampleParams(args));

    sample::gLogInfo << "Building and running a GPU inference engine for Onnx MNIST" << std::endl;

    if (!sample.build())
    {
        return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.infer())
    {
        return sample::gLogger.reportFail(sampleTest);
    }

    return sample::gLogger.reportPass(sampleTest);
}