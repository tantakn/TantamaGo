#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif



// constexpr int BOARDSIZE = 19;
constexpr int BOARDSIZE = 9;

constexpr double komi = 7.5;

constexpr bool isJapaneseRule = false;
// constexpr bool isJapaneseRule = true;

constexpr ll debugFlag = 0;
// constexpr ll debugFlag = ll(1)<<25;
// constexpr ll debugFlag = ll(1)<<31 | ll(1)<<29 | ll(1)<<30;

// const string tensorRTModelPath = "./19ro.onnx";
// const string tensorRTModelPath = "./test19_2.onnx";
const string tensorRTModelPath = "./q50k_DualNet_256_24.onnx";
// const string tensorRTModelPath = "./test9_2.onnx";


const string GPTALPHABET("ABCDEFGHJKLMNOPQRST");
const string GPTAlapabet("abcdefghjklmnopqrst");