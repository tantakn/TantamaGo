#!/bin/bash

# コンパイルオプション
CXX="/usr/bin/g++"
CXXFLAGS="-std=c++2a -fdiagnostics-color=always -g -O0 -W"
LDFLAGS="-fopenmp"

# ビルド対象のファイル
SOURCE_FILE="$1"
OUTPUT_FILE="${SOURCE_FILE%.*}"

# ビルドコマンド
$CXX $CXXFLAGS $SOURCE_FILE -o $OUTPUT_FILE $LDFLAGS

# ビルドが成功した場合に実行
if [ $? -eq 0 ]; then
    echo "ビルドが成功しました。実行します。"
    ./$OUTPUT_FILE
else
    echo "ビルドに失敗しました。"
fi