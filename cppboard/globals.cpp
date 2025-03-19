#ifndef myMacro_hpp_INCLUDED
#include "../myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif


#include "config_bunkatu.hpp"

#define dbg_flag////////////////////
#ifdef dbg_flag
ll g_node_cnt = 0;
int endCnt = 0;
int deepestMoveCnt = 0;
int expandCnt = 0;
#endif

vector<vector<char>> tmpBoard = []()
{
    vector<vector<char>> tmpBoard(BOARDSIZE + 2, vector<char>(BOARDSIZE + 2, 0b0));
    rep (i, BOARDSIZE + 2) {
        tmpBoard[0][i] = 0b11;
        tmpBoard[BOARDSIZE + 1][i] = 0b11;
        tmpBoard[i][0] = 0b11;
        tmpBoard[i][BOARDSIZE + 1] = 0b11;
    }
    return tmpBoard;
}();

/// @brief 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁
const vector<vector<char>> rawBoard = tmpBoard;

vector<vector<int>> tmpIdBoard = []() 
{
    vector<vector<int>> tmpIdBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0));
    rep (i, BOARDSIZE + 2) {
        tmpIdBoard[0][i] = -1;
        tmpIdBoard[BOARDSIZE + 1][i] = -1;
        tmpIdBoard[i][0] = -1;
        tmpIdBoard[i][BOARDSIZE + 1] = -1;
    }
    return tmpIdBoard;
}();
/// @brief 0は空点、-1は壁
const vector<vector<int>> rawIdBoard = tmpIdBoard;
