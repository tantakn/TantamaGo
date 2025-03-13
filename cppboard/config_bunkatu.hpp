// #ifndef myMacro_hpp_INCLUDED
// #include "myMacro.hpp"
// #define myMacro_hpp_INCLUDED
// #endif

// #define dbg_flag////////////////////
// #ifdef dbg_flag
// extern ll g_node_cnt = 0;
// extern int endCnt = 0;
// extern int deepestMoveCnt = 0;
// extern int expandCnt = 0;
// #endif


constexpr int BOARDSIZE = 9;
const string tensorRTModelPath = "./9_20250303_225555_370_DualNet_256_24.onnx";
// const string tensorRTModelPath = "./q50k_DualNet_256_24.onnx";
// const string tensorRTModelPath = "./test9_2.onnx";

// constexpr int BOARDSIZE = 13;
// // const string tensorRTModelPath = "./13_1_DualNet_256_24.onnx";
// const string tensorRTModelPath = "./13_20250304_005752_240_DualNet_256_24.onnx";


// constexpr int BOARDSIZE = 19;
// const string tensorRTModelPath = "./19ro.onnx";
// // const string tensorRTModelPath = "./test19_2.onnx";



constexpr double komi = 7.0;

constexpr bool isJapaneseRule = false;
// constexpr bool isJapaneseRule = true;

constexpr ll debugFlag = 1<<5;
// constexpr ll debugFlag = ll(1)<<25;
// constexpr ll debugFlag = ll(1)<<31 | ll(1)<<29 | ll(1)<<30;



const string GPTALPHABET("ABCDEFGHJKLMNOPQRST");
const string GPTAlapabet("abcdefghjklmnopqrst");



const float PUCT_C_BASE = 20403.9803;
const float PUCT_C_INIT = 0.70598003;

const float PUCB_SECOND_TERM_WEIGHT = 1.0;

constexpr bool IS_PUCT = true;

constexpr int visitMax = 10000000;


const vector<pair<char, char>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};


/// @brief 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁
extern const vector<vector<char>> rawBoard;


/// @brief 0は空点、-1は壁
extern const vector<vector<int>> rawIdBoard;