#!/bin/bash

# # コンパイルオプション
# CXX="/usr/bin/g++"
# CXXFLAGS="-std=c++2a -fdiagnostics-color=always -g -O0 -W"
# LDFLAGS=""
# # LDFLAGS="-fopenmp"

# # ビルド対象のファイル
# SOURCE_FILE="$1"
# OUTPUT_FILE="${SOURCE_FILE%.*}"

# ビルドコマンド
g++ -Wall -Wno-deprecated-declarations -std=c++17   -I"./TensorRT/common"   -I"./TensorRT/utils"   -I"./TensorRT"   -I"/usr/local/cuda/include"   -I"./TensorRT/include"   -D_REENTRANT -DTRT_STATIC=0   -g  goBoard.cpp   ./TensorRT/common/bfloat16.cpp   ./TensorRT/common/getOptions.cpp   ./TensorRT/common/logger.cpp   ./TensorRT/common/sampleDevice.cpp   ./TensorRT/common/sampleEngines.cpp   ./TensorRT/common/sampleInference.cpp   ./TensorRT/common/sampleOptions.cpp   ./TensorRT/common/sampleReporting.cpp   ./TensorRT/common/sampleUtils.cpp   ./TensorRT/utils/fileLock.cpp   ./TensorRT/utils/timingCache.cpp   -o tensorRTIgo   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/lib64"   -L"./TensorRT/lib"   -Wl,-rpath-link="./TensorRT/lib"   -L"./TensorRT/bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax

# ビルドが成功した場合に実行
if [ $? -eq 0 ]; then
    echo "ビルドが成功しました。実行します。"
    ./tensorRTIgo
else
    echo "ビルドに失敗しました。"
fi
