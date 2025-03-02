#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

#define dbg_flag////////////////////
#ifdef dbg_flag
ll g_node_cnt = 0;
int endCnt = 0;
int deepestMoveCnt = 0;
#endif


// constexpr int BOARDSIZE = 19;
// constexpr int BOARDSIZE = 13;
constexpr int BOARDSIZE = 9;

constexpr double komi = 7.0;

constexpr bool isJapaneseRule = false;
// constexpr bool isJapaneseRule = true;

constexpr ll debugFlag = 1<<5;
// constexpr ll debugFlag = ll(1)<<25;
// constexpr ll debugFlag = ll(1)<<31 | ll(1)<<29 | ll(1)<<30;



// const string tensorRTModelPath = "./19ro.onnx";
// const string tensorRTModelPath = "./test19_2.onnx";
// const string tensorRTModelPath = "./13_1_DualNet_256_24.onnx";
const string tensorRTModelPath = "./q50k_DualNet_256_24.onnx";
// const string tensorRTModelPath = "./test9_2.onnx";


const string GPTALPHABET("ABCDEFGHJKLMNOPQRST");
const string GPTAlapabet("abcdefghjklmnopqrst");



const float PUCT_C_BASE = 20403.9803;
const float PUCT_C_INIT = 0.70598003;

const float PUCB_SECOND_TERM_WEIGHT = 1.0;

constexpr bool IS_PUCT = true;

constexpr int visitMax = 10000000;