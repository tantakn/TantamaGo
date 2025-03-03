// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard/TensorRT$ tree -L 2
// .
// ├── bin
// │   ├── chobj
// │   ├── dchobj
// │   ├── sample_onnx_igo
// │   ├── sample_onnx_mnist
// │   ├── sample_onnx_mnist_debug
// │   └── trtexec
// ├── common
// │   ├── BatchStream.h
// │   ├── EntropyCalibrator.h
// │   ├── ErrorRecorder.h
// │   ├── argsParser.h
// │   ├── bfloat16.cpp
// │   ├── bfloat16.h
// │   ├── buffers.h
// │   ├── common.h
// │   ├── dumpTFWts.py
// │   ├── getOptions.cpp
// │   ├── getOptions.h
// │   ├── getopt.c
// │   ├── getoptWin.h
// │   ├── half.h
// │   ├── logger.cpp
// │   ├── logger.h
// │   ├── logging.h
// │   ├── parserOnnxConfig.h
// │   ├── safeCommon.h
// │   ├── sampleConfig.h
// │   ├── sampleDevice.cpp
// │   ├── sampleDevice.h
// │   ├── sampleEngines.cpp
// │   ├── sampleEngines.h
// │   ├── sampleEntrypoints.h
// │   ├── sampleInference.cpp
// │   ├── sampleInference.h
// │   ├── sampleOptions.cpp
// │   ├── sampleOptions.h
// │   ├── sampleReporting.cpp
// │   ├── sampleReporting.h
// │   ├── sampleUtils.cpp
// │   ├── sampleUtils.h
// │   └── streamReader.h
// ├── data
// │   ├── char-rnn
// │   ├── int8_api
// │   ├── mnist
// │   └── resnet50
// ├── include
// │   ├── NvInfer.h
// │   ├── NvInferImpl.h
// │   ├── NvInferLegacyDims.h
// │   ├── NvInferPlugin.h
// │   ├── NvInferPluginBase.h
// │   ├── NvInferPluginUtils.h
// │   ├── NvInferRuntime.h
// │   ├── NvInferRuntimeBase.h
// │   ├── NvInferRuntimeCommon.h
// │   ├── NvInferRuntimePlugin.h
// │   ├── NvInferVersion.h
// │   ├── NvOnnxConfig.h
// │   └── NvOnnxParser.h
// ├── lib
// │   ├── libnvinfer.so -> libnvinfer.so.10.7.0
// │   ├── libnvinfer.so.10 -> libnvinfer.so.10.7.0
// │   ├── libnvinfer.so.10.7.0
// │   ├── libnvinfer_builder_resource.so.10.7.0
// │   ├── libnvinfer_builder_resource_win.so.10.7.0
// │   ├── libnvinfer_dispatch.so -> libnvinfer_dispatch.so.10.7.0
// │   ├── libnvinfer_dispatch.so.10 -> libnvinfer_dispatch.so.10.7.0
// │   ├── libnvinfer_dispatch.so.10.7.0
// │   ├── libnvinfer_dispatch_static.a
// │   ├── libnvinfer_lean.so -> libnvinfer_lean.so.10.7.0
// │   ├── libnvinfer_lean.so.10 -> libnvinfer_lean.so.10.7.0
// │   ├── libnvinfer_lean.so.10.7.0
// │   ├── libnvinfer_lean_static.a
// │   ├── libnvinfer_plugin.so -> libnvinfer_plugin.so.10.7.0
// │   ├── libnvinfer_plugin.so.10 -> libnvinfer_plugin.so.10.7.0
// │   ├── libnvinfer_plugin.so.10.7.0
// │   ├── libnvinfer_plugin_static.a
// │   ├── libnvinfer_static.a
// │   ├── libnvinfer_vc_plugin.so -> libnvinfer_vc_plugin.so.10.7.0
// │   ├── libnvinfer_vc_plugin.so.10 -> libnvinfer_vc_plugin.so.10.7.0
// │   ├── libnvinfer_vc_plugin.so.10.7.0
// │   ├── libnvinfer_vc_plugin_static.a
// │   ├── libnvonnxparser.so -> libnvonnxparser.so.10
// │   ├── libnvonnxparser.so.10 -> libnvonnxparser.so.10.7.0
// │   ├── libnvonnxparser.so.10.7.0
// │   ├── libnvonnxparser_static.a
// │   ├── libonnx_proto.a
// │   └── stubs
// └── utils
//     ├── fileLock.cpp
//     ├── fileLock.h
//     ├── timingCache.cpp
//     └── timingCache.h

// 14 directories, 82 files

// print が衝突してたからbufferなんとかのを変えたはず。標準出力を抑制もしたはず。
// TnsorRT/common/buffers.h の void printBuffer(std::ostream& os, void* buf, size_t bufSize, size_t rowCount) を void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount) に変更。


// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ g++ -w -Wno-deprecated-declarations -std=c++17   -I"./TensorRT/common"   -I"./TensorRT/utils"   -I"./TensorRT"   -I"/usr/local/cuda/include"   -I"./TensorRT/include"   -D_REENTRANT -DTRT_ST ATIC=0   -g  goBoard.cpp   ./TensorRT/common/bfloat16.cpp   ./TensorRT/common/getOptions.cpp   ./TensorRT/common/logger.cpp   ./TensorRT/common/sampleDevice.cpp   ./TensorRT/common/sampleEngines.cpp   ./TensorRT/common/sampleInference.cpp   ./TensorRT/common/sampleOptions.cpp   ./TensorRT/common/sampleReporting.cpp   ./TensorRT/common/sampleUtils.cpp   ./TensorRT/utils/fileLock.cpp   ./TensorRT/utils/timingCache.cpp   -o goBoard   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/ lib64"   -L"./TensorRT/lib"   -Wl,-rpath-link="./TensorRT/lib"   -L"./TensorRT/bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax

// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ ./goBoard 2>&1 | tee -a ../zzlog/`date '+%Y%m%d_%H%M%S'`goBoard.txt




#ifndef myMacro_hpp_INCLUDED
#include "../myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif



#ifndef config_hpp_INCLUDED
#include "config.hpp"
#define config_hpp_INCLUDED
#endif



#ifndef tensorRTigo_cpp_INCLUDED
#include "./tensorRTigo.cpp"
#define tensorRTigo_cpp_INCLUDED
#endif

#ifndef goBoard_hpp_INCLUDED
#include "goBoard.hpp"
#define goBoard_hpp_INCLUDED
#endif

// ソケット通信用
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>


mt19937 mt(random_device{}());

// goBoard* rootPtr = nullptr; //rootPtrグローバル廃止

goBoard::goBoard()
    : board(rawBoard), idBoard(rawIdBoard), teban(1), parent(nullptr), isRoot(true), moveCnt(0)
{
    // assert(rootPtr == nullptr); //rootPtrグローバル廃止
    // rootPtr = this;

    libs[-1] = INF;
}

goBoard::goBoard(vector<vector<char>> inputBoard, char inputTeban)
    : board(InputBoardFromVec(inputBoard)), idBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0)), teban(inputTeban), isRoot(true), moveCnt(0)
{
    /// TODO: teban の扱いを考える

    // assert(rootPtr == nullptr); //rootPtrグローバル廃止
    // rootPtr = this;

    rep (i, BOARDSIZE + 2) {
        idBoard[0][i] = -1;
        idBoard[BOARDSIZE + 1][i] = -1;
        idBoard[i][0] = -1;
        idBoard[i][BOARDSIZE + 1] = -1;

        libs[-1] = INF;
    }

    int cnt = 1;
    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (idBoard[i][j] == 0 && board[i][j] != 0) {
                ApplyString(i, j);
            }
        }
    }
}

goBoard::goBoard(goBoard& inputparent, int y, int x, char putcolor)
    : parent(&inputparent), board(inputparent.board), idBoard(inputparent.idBoard), libs(inputparent.libs), stringIdCnt(inputparent.stringIdCnt), history(inputparent.history), teban(1), previousMove(make_pair(y, x)), isRoot(false), moveCnt(inputparent.moveCnt + 1)
{
    /// TODO: putcolor 要る？

    assert((x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE && (putcolor == 0b01 || putcolor == 0b10)) || (x == 0 && y == 0));

    if (x == 0 && y == 0) {
        if (parent->previousMove == make_pair((char)0, (char)0)) {
            isEnded = true;
            teban = 3 - parent->teban;
            previousMove = make_pair(0, 0);
            return;
        }

        teban = 3 - parent->teban;
        previousMove = make_pair(0, 0);
        return;
    }

    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (libs[idBoard[ny][nx]] == 1 && board[ny][nx] == 3 - putcolor) {
            DeleteString(ny, nx);
        }
    }

    board[y][x] = putcolor;

    ApplyString(y, x);

    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (idBoard[ny][nx] >= 1) {
            ApplyString(ny, nx);
        }
    }

    teban = 3 - putcolor;

    history.insert(board);

    parent->childrens[previousMove] = this;
}

goBoard::~goBoard()
{
    // for (auto& x : childrens) とかだと壊れるみたい
    vector<tuple<char, char, goBoard*>> tmpChildrens;
    for (pair<pair<char, char>, goBoard*> x : childrens) {
        tmpChildrens.push_back(make_tuple(x.first.first, x.first.second, x.second));
    }
    for (auto [y, x, p] : tmpChildrens) {
        delete p;
    }

    if (!isRoot) {
        parent->childrens.erase(previousMove);
    }
}

goBoard* goBoard::SucceedRoot(goBoard*& rootPtr, pair<char, char> move)
{
    assert(this->isRoot);
    assert(rootPtr == this);
    // assert(this->childrens.count(move));

    if (!childrens.count(move)) {
        PutStone(move.first, move.second, teban);
    }


    goBoard* tmp = this->childrens[move];
    this->childrens.erase(move);
    tmp->parent = nullptr;
    tmp->isRoot = true;
    rootPtr = tmp;

    delete this;

    return tmp;
}


tuple<int, float, float, float> goBoard::ExpandNode(TensorRTOnnxIgo tensorRT)
{
    assert(childrens.size() == 0);
    assert(!isEnded);


    // ++numVisits;

    vector<tuple<char, char, char>> legalMoves = GenAllLegalMoves();

    numVisits = legalMoves.size();

    /// 推論の結果を一時保存する配列。tmpPolicy[BOARDSIZE * BOARDSIZE] はパス。
    vector<float> tmpPolicy(BOARDSIZE * BOARDSIZE + 1, 0.0);


    tensorRT.infer(MakeInputPlane(), tmpPolicy, values);



    for (auto [y, x, t] : legalMoves) {
        if (x == 0 && y == 0) {
            float tmp = tmpPolicy[BOARDSIZE * BOARDSIZE];
            policys[make_pair(y, x)] = tmp;
            continue;
        }
        float tmp = tmpPolicy[(y - 1) * BOARDSIZE + x - 1];
        policys[make_pair(y, x)] = tmp;
    }



    if (0) {  // softmaxなし
        for (auto [move, x] : policys) {
            float tmpUct;
            if (IS_PUCT) {
                tmpUct = x + PUCB_SECOND_TERM_WEIGHT * sqrt(log(policys.size())) / 2;
            }
            else {
                tmpUct = x + sqrt(2 * log(policys.size()));
            }
            lock_guard<mutex> lock(uctsMutex);
            ucts.insert(make_tuple(tmpUct, 1, x, move));
        }

        if (debugFlag & 1 << 31) {
            PrintBoard(1 << 31);
        }

        return tie(teban, values[0], values[1], values[2]);
    }
    else if (1) {  // softmaxあり
        // softmaxで使う変数
        float maxPolicy = 0.0;

        for (auto [y, x, t] : legalMoves) {
            if (x == 0 && y == 0) {
                float tmp = tmpPolicy[BOARDSIZE * BOARDSIZE];
                policys[make_pair(y, x)] = tmp;
                chmax(maxPolicy, tmp);
                continue;
            }
            float tmp = tmpPolicy[(y - 1) * BOARDSIZE + x - 1];
            policys[make_pair(y, x)] = tmp;
            chmax(maxPolicy, tmp);
        }


        // valueにsoftmax
        float bunbo = 0.0;
        for (auto x : values) {
            bunbo += exp(x);
        }

        for (auto& x : values) {
            x = exp(x) / bunbo;
        }

        // tmppolicyにsoftmax
        map<std::pair<char, char>, float> tmpPolicys;
        bunbo = 0.0;
        for (auto [move, x] : policys) {
            bunbo += exp(x - maxPolicy);
        }

        float maxPolicy2 = 0.0;
        for (auto [move, x] : policys) {
            tmpPolicys[move] = exp(x - maxPolicy) / bunbo;
            chmax(maxPolicy2, tmpPolicys[move]);
        }


        // tmpPolicys の最大が values[2] + values[1] * 0.5 になるように調整
        for (auto [move, x] : tmpPolicys) {
            tmpPolicys[move] = (values[2] + values[1] * 0.5) * x / maxPolicy2;
            // policys[move] = values[2] * x / maxPolicy2;
            if (move == make_pair(char(0), char(0))) {
                tmpPolicys[move] -= 0.5;
            }
        }
        // policyにsoftmaxここまで


        // // policyにsoftmax
        // bunbo = 0.0;
        // for (auto [move, x] : policys) {
        //     bunbo += exp(x - maxPolicy);
        // }

        // float maxPolicy2 = 0.0;
        // for (auto [move, x] : policys) {
        //     policys[move] = exp(x - maxPolicy) / bunbo;
        //     chmax(maxPolicy2, policys[move]);
        // }


        // // policys の最大が values[2] + values[1] * 0.5 になるように調整
        // for (auto [move, x] : policys) {
        //     policys[move] = (values[2] + values[1] * 0.5) * x / maxPolicy2;
        //     // policys[move] = values[2] * x / maxPolicy2;
        //     if (move == make_pair(char(0), char(0))) {
        //         policys[move] -= 0.5;
        //     }
        // }
        // // policyにsoftmaxここまで


        for (auto [move, x] : policys) {
            float tmpUct;
            if (IS_PUCT) {
                tmpUct = x + PUCB_SECOND_TERM_WEIGHT * sqrt(log(policys.size())) / 2;
            }
            else {
                tmpUct = x + sqrt(2 * log(policys.size()));
            }
            lock_guard<mutex> lock(uctsMutex);
            // ucts.insert(make_tuple(tmpUct, 1, x, move));
            ucts.insert(make_tuple(tmpUct, 1, tmpPolicys[move], move));
        }

        if (debugFlag & 1 << 31) {
            PrintBoard(1 << 31);
        }

#ifdef dbg_flag
        chmax(deepestMoveCnt, this->moveCnt);
#endif

        return tie(teban, values[0], values[1], values[2]);
    }
}


bool goBoard::UpdateUcts(tuple<int, float, float, float> input, pair<char, char> inputMove)
{
    auto [inputColor, inputLoseValue, inputDrawValue, inputWinValue] = input;

    if (inputColor != teban) {
        // if (inputColor != teban) {/////////////////////////
        swap(inputWinValue, inputLoseValue);
    }

    inputWinValue = inputWinValue + inputDrawValue * 0.5;


    lock_guard<mutex> lock(uctsMutex);

    set<tuple<double, int, float, pair<char, char>>> tmpUcts;

    ++numVisits;

    if (IS_PUCT) {
        // puct の場合
        for (auto [puct, cnt, valueSum, puctMove] : ucts) {
            if (inputMove == puctMove) {
                cnt += 1;
                valueSum += inputWinValue;
            }

            double newUct = valueSum / cnt + PUCB_SECOND_TERM_WEIGHT * policys[puctMove] * sqrt(log(numVisits)) / (1 + cnt);
            // double newUct = valueSum / cnt + PUCB_SECOND_TERM_WEIGHT * sqrt(log(cnt)) / (1 + numVisits);  // 多分逆

            tmpUcts.insert(make_tuple(newUct, cnt, valueSum, puctMove));
        }
    }
    else {
        // uct の場合
        for (auto [uct, cnt, winSum, uctMove] : ucts) {
            if (inputMove == uctMove) {
                int newCnt = cnt + 1;
                float newWinSum = winSum + inputWinValue;
                double newUct = newWinSum / newCnt + sqrt(2 * log(numVisits) / newCnt);

                tmpUcts.insert(make_tuple(newUct, newCnt, newWinSum, uctMove));

                continue;
            }

            tmpUcts.insert(make_tuple(winSum / cnt + sqrt(2 * log(numVisits) / cnt), cnt, winSum, uctMove));
        }
    }

    ucts = tmpUcts;

    return 0;
}



pair<char, char> goBoard::GetBestMove()
{
    assert(childrens.size() > 0);

    lock_guard<mutex> lock(uctsMutex);


    double maxVisit = -INF;
    pair<char, char> ans;

    for (auto [uct, visit, winSum, move] : ucts) {
        if (maxVisit < visit) {
            maxVisit = visit;
            ans = move;
        }
        else if (maxVisit == visit && policys[ans] < policys[move]) {
            ans = move;
        }
    }

    return ans;
}


vector<vector<char>> goBoard::InputBoardFromVec(vector<vector<char>> input)
{
    if (int(input.size()) == BOARDSIZE) {
        for (auto row : input) {
            if (int(row.size()) != BOARDSIZE) {
                print("int(row.size()):", int(row.size()));
                assert(false && "👺Input size is invalid");
            }
        }
    }
    else {
        print("int(input.size()):", int(input.size()));
        assert(false && "👺Input size is invalid");
    }

    vector<vector<char>> tmp = rawBoard;

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            tmp[i][j] = input[i - 1][j - 1];
        }
    }

    return tmp;
};

void goBoard::PrintBoard(ll bit = 0b1)
{
    if (bit & 0b0001) {
        // print("board: ", board);
        cerr << "previousMove: " << (int)previousMove.first << " " << (int)previousMove.second << " " << 3 - (int)teban << endl;  ////////////////
        cerr << "moveCnt: " << (int)moveCnt << ", teban: " << (int)teban << endl;
        cerr << "   " << flush;
        rep (i, BOARDSIZE) {
            cerr << ' ' << GPTALPHABET[i] << flush;
        }
        cerr << endl;
        rep (i, board.size()) {
            if (i == 0 || i == BOARDSIZE + 1) {
                continue;
            }
            cerr << setw(2) << setfill(' ') << i << " " << flush;
            rep (j, board.size()) {
                if (board[i][j] == 3) {
                }
                else if (board[i][j] == 1) {
                    cerr << " ●" << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << " ○" << flush;
                }
                else if (i == 1 && j == 1) {
                    cerr << " ┌" << flush;
                }
                else if (i == BOARDSIZE && j == 1) {
                    cerr << " └" << flush;
                }
                else if (i == 1 && j == BOARDSIZE) {
                    cerr << " ┐" << flush;
                }
                else if (i == BOARDSIZE && j == BOARDSIZE) {
                    cerr << " ┘" << flush;
                }
                else if (i == 1) {
                    cerr << " ┬" << flush;
                }
                else if (i == BOARDSIZE) {
                    cerr << " ┴" << flush;
                }
                else if (j == 1) {
                    cerr << " ├" << flush;
                }
                else if (j == BOARDSIZE) {
                    cerr << " ┤" << flush;
                }
                else {
                    cerr << " +" << flush;
                }
            }
            cerr << endl;
        }
    }

    if (bit & 0b010) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cerr << (int)board[i][j] << " " << flush;
            }
            cerr << endl;
        }
        cerr << endl;
    }

    if (bit & 0b100) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cerr << setw(3) << setfill(' ') << idBoard[i][j] << " " << flush;
            }
            cerr << endl;
        }
        cerr << endl;
    }

    if (bit & 0b1000) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(3) << setfill(' ') << libs[idBoard[i][j]] << " " << flush;
            }
            cerr << endl;
        }
        cerr << endl;
    }

    // ニューラルネットワーク入力用
    if (bit & 0b10000) {
        cerr << (int)previousMove.first << " " << (int)previousMove.second << " " << (int)teban << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << (int)board[i][j] << " " << flush;
            }
            cerr << endl;
        }
    }

    // 推論の結果。softmax後。
    if (bit & 1 << 31) {
        print("policys勝率*1000 policys.size():", policys.size());
        vector<vector<float>> tmp(BOARDSIZE + 2, vector<float>(BOARDSIZE + 2, -1000000));
        for (auto [move, x] : policys) {
            // print(move, x);
            tmp[move.first][move.second] = x;
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    if (tmp[i][j] == -1000000) {
                        cerr << "---- " << flush;
                    }
                    else if (tmp[i][j] >= 0) {
                        cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 1000) << " " << flush;
                        // cerr << fixed << setprecision(4) << tmp[i][j] << " ";
                    }
                    else {
                        cerr << setw(4) << setfill(' ') << 0 << " " << flush;
                    }
                }
                else if (board[i][j] == 1) {
                    cerr << "#### " << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ " << flush;
                }
                else {
                    cerr << "???? " << flush;
                }
                // cerr << fixed << setprecision(4) << tmp[i][j] << " ";
            }
            cerr << endl;
        }
        print("pass:", int(tmp[0][0] * 1000));  //////////////
        print("values:", values);
        cerr << resetiosflags(ios_base::floatfield);
    }

    // childの有無
    if (bit & 1 << 30) {
        print("childrens.size():", childrens.size());
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (childrens.count(make_pair(i, j))) {
                    cerr << "O " << flush;
                }
                else {
                    cerr << "X " << flush;
                }
            }
            cerr << endl;
        }
    }

    // uctの表示
    if (bit & 1 << 29) {
        lock_guard<mutex> lock(uctsMutex);

        cerr << "uct値*100 を表示 ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, -1000000));
        int maxCnt = 0;
        pair<char, char> maxMove;
        double maxUct = -INF;
        int maxVisit = 0;
        double maxWinSum = 0;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = uct;
            if (cnt > maxCnt) {
                maxCnt = cnt;
                maxMove = move;
                maxUct = uct;
                maxVisit = cnt;
                maxWinSum = winSum;
            }
            /// TODO: uct ではなく勝率で比較したい
            else if (cnt == maxCnt && uct > tmp[maxMove.first][maxMove.second]) {
                maxCnt = cnt;
                maxMove = move;
                maxUct = uct;
                maxVisit = cnt;
                maxWinSum = winSum;
            }
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    if (tmp[i][j] == -1000000) {
                        cerr << "---- " << flush;
                    }
                    else if (tmp[i][j] >= 0) {
                        cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 100) << " " << flush;
                        // cerr << fixed << setprecision(1) << showpoint << tmp[i][j] * 10 << " ";
                    }
                    else {
                        cerr << setw(4) << setfill(' ') << 0 << " " << flush;
                    }
                }
                else if (board[i][j] == 1) {
                    cerr << "#### " << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ " << flush;
                }
                else {
                    cerr << "???? " << flush;
                }
            }
            cerr << endl;
        }
        cerr << "pass: " << int(tmp[0][0] * 100) << endl;
        cerr << "ans: " << int(this->teban) << " [" << int(maxMove.first) << ", " << int(maxMove.second) << "]" << ", uct: " << maxUct << ", visit: " << maxVisit << ", winrate:" << maxWinSum / maxVisit << endl;
    }

    // visitの表示
    if (bit & 1 << 28) {
        lock_guard<mutex> lock(uctsMutex);

        cerr << "visitの表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        vector<vector<int>> tmp(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, -1000000));
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = cnt;
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    if (tmp[i][j] == -1000000) {
                        cerr << "---- " << flush;
                    }
                    else {
                        cerr << setw(4) << setfill(' ') << tmp[i][j] << " " << flush;
                    }
                }
                else if (board[i][j] == 1) {
                    cerr << "#### " << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ " << flush;
                }
                else {
                    cerr << "???? " << flush;
                }
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ') << tmp[0][0] << endl;
    }

    // 勝率の表示
    if (bit & 1 << 27) {
        lock_guard<mutex> lock(uctsMutex);

        cerr << "探索後勝率*1000 の表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
#ifdef dbg_flag
        cout << "endCnt: " << endCnt << ", depth: " << deepestMoveCnt - this->moveCnt << endl;
#endif
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, -1000000));
        pair<char, char> maxMove;
        double maxWinRate = -INF;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = winSum / cnt;
            if (winSum / cnt >= maxWinRate) {
                maxWinRate = winSum / cnt;
                maxMove = move;
            }
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    if (tmp[i][j] == -1000000) {
                        cerr << "---- " << flush;
                    }
                    else if (tmp[i][j] >= 0) {
                        cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 1000) << " " << flush;
                    }
                    else {
                        cerr << setw(4) << setfill(' ') << 0 << " " << flush;
                    }
                }
                else if (board[i][j] == 1) {
                    cerr << "#### " << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ " << flush;
                }
                else {
                    cerr << "???? " << flush;
                }
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ') << int(tmp[0][0] * 1000) << endl;
        print("maxWinRate:", maxWinRate, maxMove);
        print("infervalues:", values);
    }

    // ペナルティの表示
    if (bit & 1 << 25) {
        lock_guard<mutex> lock(uctsMutex);

        cerr << "ペナルティ*100 の表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, 0));
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = sqrt(2 * log(numVisits) / (cnt));
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 100) << " " << flush;
                }
                else if (board[i][j] == 1) {
                    cerr << "#### " << flush;
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ " << flush;
                }
                else {
                    cerr << "???? " << flush;
                }
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ') << int(tmp[0][0] * 100) << endl;
    }

    // 合法手を表示（0が合法手)
    if (bit & 1 << 26) {
        if (isEnded) {
            return;
        }
        cerr << "合法手の表示。teban: " << int(this->teban) << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(4) << setfill(' ') << IsIllegalMove(i, j, teban) << " " << flush;
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ') << IsIllegalMove(0, 0, teban) << endl;
    }


    cerr << resetiosflags(std::ios::floatfield);  // 浮動小数点の書式をリセット
    cerr << resetiosflags(std::ios::showpoint);   // showpoint をリセット
    cerr << resetiosflags(std::ios::showpos);     // showpos をリセット
    cerr << std::defaultfloat;
};

string goBoard::ToJson()
{
    vector<vector<char>> v = board;
    vector<char> tmp = {teban, previousMove.first, previousMove.second};
    v.push_back(tmp);
    nlohmann::json j(v);

    return j.dump();
}

int goBoard::CountLiberties(int y, int x)
{
    assert(x >= 0 && x <= BOARDSIZE + 1 && y >= 0 && y <= BOARDSIZE + 1);

    if (board[y][x] == 0) {
        return -1;
    };

    if (x == 0 || x == BOARDSIZE + 1 || y == 0 || y == BOARDSIZE + 1) {
        return INF;
    }

    int color = board[y][x];

    vector<vector<char>> boardSearched = rawBoard;
    boardSearched[y][x] = 1;

    queue<pair<int, int>> bfs;

    bfs.push({x, y});

    int cnt = 0;

    while (!bfs.empty()) {
        auto [x, y] = bfs.front();
        bfs.pop();
        boardSearched[y][x] = 1;

        for (auto dir : directions) {
            int nx = x + dir.first;
            int ny = y + dir.second;

            if (boardSearched[ny][nx]) {
                continue;
            }
            else if (board[ny][nx] == 0) {
                boardSearched[ny][nx] = 1;
                ++cnt;
                continue;
            }
            else if (board[ny][nx] == color) {
                bfs.push({nx, ny});
            }
        }
    }
    return cnt;
};

int goBoard::IsIllegalMove(int y, int x, char color)
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE && (color == 0b01 || color == 0b10) || (x == 0 && y == 0));

    if (isEnded) {
        return 5;
    }

    if (x == 0 && y == 0) {
        return 0;
    }

    // すでに石がある場合はfalse
    if (board[y][x] != 0) {
        return 1;
    }

    // 自殺手 || 超コウルール で合法手でない

    // 打つ場所の4方が囲まれている && 囲っている自分の石の呼吸点がすべて1 && !囲っている相手の石に呼吸点が1のものがある なら自殺手

    // 超コウルールは取れる石を取ってから判定する

    bool isSuicide = true;
    set<int> toTakeEnemyIds;
    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (board[ny][nx] == 0) {
            isSuicide = false;
        }
        else if (board[ny][nx] == color) {
            if (libs[idBoard[ny][nx]] != 1) {
                isSuicide = false;
            }
        }
        else if (board[ny][nx] == 3 - color) {
            if (libs[idBoard[ny][nx]] == 1) {
                isSuicide = false;
                toTakeEnemyIds.insert(idBoard[ny][nx]);
            }
        }
    }
    if (isSuicide) {
        return 2;
    }

    // 超コウルール
    vector<vector<char>> tmp = board;

    for (auto id : toTakeEnemyIds) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (idBoard[i][j] == id) {
                    tmp[i][j] = 0;
                }
            }
        }
    }
    tmp[y][x] = color;
    if (history.count(tmp)) {
        return 3;
    }

    // 終局のために、2眼以上ある石の目を埋める手に良い手が無いと仮定して、その手を禁止とする。
    // とりあえず、四方を同じ連が囲っている場合に眼として、その眼を埋める手を禁止とする。

    // // 何故か動かない
    // bool isFillEye = true;
    // char tmpId = -2;
    // for (auto dir : directions) {
    //     int nx = x + dir.first;
    //     int ny = y + dir.second;

    //     if (board[ny][nx] == 0) {
    //         isFillEye = false;
    //         break;
    //     }

    //     if (board[ny][nx] == 3) {
    //         continue;
    //     }

    //     if (tmpId == -2) {
    //         tmpId = idBoard[ny][nx];
    //         continue;
    //     }

    //     if (tmpId != idBoard[ny][nx]) {
    //         isFillEye = false;
    //         break;
    //     }
    // }

    // if (isFillEye) {
    //     return 4;
    // }

    return 0;
};


vector<tuple<char, char, char>> goBoard::GenAllLegalMoves()
{
    assert(!isEnded);

    vector<tuple<char, char, char>> legalMoves(0);

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (!IsIllegalMove(i, j, teban)) {
                legalMoves.push_back({i, j, teban});
            }
        }
    }

    legalMoves.push_back({0, 0, 0});

    return legalMoves;
}


void goBoard::ApplyString(int y, int x)
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE);

    if (board[y][x] == 0) return;

    ++stringIdCnt;

    char color = board[y][x];

    /// TODO: 2つの string をつなげるとき、それぞれの lib - 1 を足し合わせればいい？
    /// TODO: lib == 0 の string で assert 出したい

    int lib = CountLiberties(y, x);
    libs[stringIdCnt] = lib;

    queue<pair<int, int>> bfs;

    bfs.push({x, y});

    while (!bfs.empty()) {
        auto [x, y] = bfs.front();
        bfs.pop();

        idBoard[y][x] = stringIdCnt;

        for (auto dir : directions) {
            char nx = x + dir.first;
            char ny = y + dir.second;

            if (board[ny][nx] == color && idBoard[ny][nx] != stringIdCnt) {
                bfs.push({nx, ny});
            }
        }
    }
}

void goBoard::DeleteString(int y, int x)
{
    assert(x >= 1 && x <= BOARDSIZE + 1 && y >= 1 && y <= BOARDSIZE + 1);
    assert(idBoard[y][x] != 0);
    assert(board[y][x] != 0);
    assert(libs[idBoard[y][x]] != 0);

    int id = idBoard[y][x];
    libs[id] = 0;

    vector<int> neighborIds(0);

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (idBoard[i][j] == id) {
                board[i][j] = 0;
                idBoard[i][j] = 0;

                for (auto dir : directions) {
                    char nx = j + dir.first;
                    char ny = i + dir.second;

                    if (idBoard[ny][nx] != 0 && idBoard[ny][nx] != id) {
                        neighborIds.push_back(idBoard[ny][nx]);
                    }
                }
            }
        }
    }

    for (auto neighborId : neighborIds) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (idBoard[i][j] == neighborId) {
                    libs[neighborId] = CountLiberties(i, j);
                }
            }
        }
    }
}

goBoard* goBoard::PutStone(int y, int x, char color)
{
    /// TODO: parent とか children, history, libBoard とかの処理も書く。goBoard はポインタで返したほうがいい？
    /// TODO: goBoard はポインタで返したほうがいい？
    /// TODO: string の処理を考える


    assert(!isEnded);
    if (IsIllegalMove(y, x, color)) {  ///////////////
        print(y, x, color);
        print(IsIllegalMove(y, x, color));
        this->PrintBoard(0b1111);
    }
    assert(!IsIllegalMove(y, x, color));

    if (childrens.count(make_pair(y, x))) {
        return childrens[make_pair(y, x)];
    }

    goBoard* p = new goBoard(*this, y, x, color);
    childrens[make_pair(y, x)] = p;

    if (debugFlag & 0b10) p->PrintBoard();

    return p;
};

tuple<char, char, char> goBoard::GenRandomMove()
{
    if (isEnded) {
        return {-1, -1, -1};
    }

    vint v(BOARDSIZE * BOARDSIZE + 1);
    rep (i, BOARDSIZE * BOARDSIZE + 1) {
        v[i] = i;
    }
    shuffle(v.begin(), v.end(), mt);

    for (int i : v) {
        if (i == BOARDSIZE * BOARDSIZE) {
            // return {1, 1, 0};
            continue;
        }
        int x = i % BOARDSIZE + 1;
        int y = i / BOARDSIZE + 1;
        if (!IsIllegalMove(y, x, teban)) {
            return {y, x, teban};
        }
        else if (debugFlag & 0b100) {
            print(x, y, IsIllegalMove(y, x, teban));
        }
    }

    // パス
    return {0, 0, 0};
};

double goBoard::CountResult()
{
    /// TODO: 日本ルール用の暫定措置。どうにかしたい。白黒が隣り合っているところでラインを引いて地を数える？中国ルールで最後までプレイしてみる？
    if (1) {  ////////////////////
              // if (isJapaneseRule) {
        double tmpBlackScore;
        if (values.size()) {
            if (values[0] > values[2] && values[0] > values[1]) {
                tmpBlackScore = -1.0;
            }
            else if (values[1] > values[0] && values[1] > values[2]) {
                tmpBlackScore = 0.0;
            }
            else {
                tmpBlackScore = 1.0;
            }

            if (teban == 1) {
                return tmpBlackScore;
            }
            else {
                return tmpBlackScore * -1;
            }
        }
        else if (parent->values.size()) {
            if (parent->values[0] > parent->values[2] && parent->values[0] > parent->values[1]) {
                tmpBlackScore = -1.0;
            }
            else if (parent->values[1] > parent->values[0] && parent->values[1] > parent->values[2]) {
                tmpBlackScore = 0.0;
            }
            else {
                tmpBlackScore = 1.0;
            }

            if (teban == 1) {
                return tmpBlackScore;
            }
            else {
                return tmpBlackScore * -1;
            }
        }
        else {
            assert(false && "👺values がない");
            return 0;
        }
    }


    /// TODO: セキは？味方の連の呼吸点が1になるような手は打たなくても良いことにする？
    /// TODO:

    int blackScore = 0;
    int whiteScore = 0;


    // vector<vector<char>> count_board = rawBoard;
    // vector<vector<char>> visited_board = rawBoard;
    // auto saiki = [&](auto self, int y, int x, char color)
    // {
    //     if (count_board[y][x] != 0) {
    //         return count_board[y][x];
    //     }

    //     if (board[y][x] == 1 || board[y][x] == 2) {
    //         if (libs[idBoard[y][x]] >= 2) {
    //             return board[y][x];
    //         }
    //         else {
    //             return char(-1);
    //         }
    //     }
    //     else if (board[y][x] == 3) {
    //         return char(-1);
    //     }


    //     count_board[y][x] = -1;


    //     char tmp = -1;
    //     for (auto dir : directions) {
    //         int nx = x + dir.first;
    //         int ny = y + dir.second;

    //         char tmp2 = self(ny, nx, color);

    //         if (tmp == -1) {
    //             tmp = tmp2;
    //         }
    //         else if (tmp2 == -1) {

    //         }
    //         else if (tmp != tmp2) {
    //             tmp = 4;
    //         }
    //     }

    //     count_board[y][x] = tmp;
    //     return tmp;
    // };

    // return teban * 2 - 3 のところは、中国ルールで最後まで埋めていない場合、2回目にパスした側が負けたことにするための処理。
    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (board[i][j] == 0) {
                /// TODO: 全部見る必要はない
                char tmpColor = 0;
                for (auto dir : directions) {
                    int nx = j + dir.first;
                    int ny = i + dir.second;
                    if (board[ny][nx] == 1) {
                        if (tmpColor == 2 && !IsIllegalMove(i, j, teban)) {
                            return (teban * 2 - 3) / 10;
                        }
                        // assert(tmpColor != 2);

                        tmpColor = 1;
                    }
                    else if (board[ny][nx] == 2) {
                        if (tmpColor == 1 && !IsIllegalMove(i, j, teban)) {
                            return (teban * 2 - 3) / 10;
                        }
                        // assert(tmpColor != 1);

                        tmpColor = 2;
                    }
                    else return (teban * 2 - 3) / 10;
                }
                if (tmpColor == 1) {
                    ++blackScore;
                }
                else if (tmpColor == 2) {
                    ++whiteScore;
                }
            }
            else if (board[i][j] == 1) {
                ++blackScore;
            }
            else if (board[i][j] == 2) {
                ++whiteScore;
            }
        }
    }
    // // return teban * 2 - 3 のところは、中国ルールで最後まで埋めていない場合、2回目にパスした側が負けたことにするための処理。
    // rep (i, 1, BOARDSIZE + 1) {
    //     rep (j, 1, BOARDSIZE + 1) {
    //         if (board[i][j] == 0) {
    //             /// TODO: 全部見る必要はない
    //             char tmpColor = 0;
    //             for (auto dir : directions) {
    //                 int nx = j + dir.first;
    //                 int ny = i + dir.second;
    //                 if (board[ny][nx] == 1) {
    //                     if (tmpColor == 2 && !IsIllegalMove(i, j, 1)) {
    //                         return (teban * 2 - 3) / 10;
    //                     }
    //                     // assert(tmpColor != 2);

    //                     tmpColor = 1;
    //                 }
    //                 else if (board[ny][nx] == 2) {
    //                     if (tmpColor == 1 && !IsIllegalMove(i, j, 1)) {
    //                         return (teban * 2 - 3) / 10;
    //                     }
    //                     // assert(tmpColor != 1);

    //                     tmpColor = 2;
    //                 }
    //                 else return (teban * 2 - 3) / 10;
    //             }
    //             if (tmpColor == 1) {
    //                 ++blackScore;
    //             }
    //             else if (tmpColor == 2) {
    //                 ++whiteScore;
    //             }
    //         }
    //         else if (board[i][j] == 1) {
    //             ++blackScore;
    //         }
    //         else if (board[i][j] == 2) {
    //             ++whiteScore;
    //         }
    //     }
    // }


    if (debugFlag & 0b100000) {
        print("blackScore:", blackScore);  ///////////
        print("whiteScore:", whiteScore);
        print("komi:", komi);
    }

    return blackScore - whiteScore - komi;
}

// bool goBoard::TestPipe() {
//     char data[bufsize] = {};

//     FILE *fp = popen("python3 ./pipetest.py", "w");
//     fputs(self.ToJson(), fp);
//     fgets(data, bufsize , fp);
//     cerr << data << endl;
//     pclose(fp);

//     return(0);
// }

vector<vector<vector<float>>> goBoard::MakeInputPlane()
{
    vector<vector<vector<float>>> inputPlane(6, vector<vector<float>>(BOARDSIZE, vector<float>(BOARDSIZE, 0)));

    rep (i, BOARDSIZE) {
        rep (j, BOARDSIZE) {
            if (board[i + 1][j + 1] == 0) {
                inputPlane[0][i][j] = 1;
            }
            else if (board[i + 1][j + 1] == teban) {
                inputPlane[1][i][j] = 1;
            }
            else if (board[i + 1][j + 1] != teban) {
                inputPlane[2][i][j] = 1;
            }
        }
    }

    if (previousMove.first != -1 && previousMove.first != 0) {
        inputPlane[3][previousMove.first - 1][previousMove.second - 1] = 1;
    }
    else if (previousMove.first == 0) {
        rep (i, BOARDSIZE) {
            rep (j, BOARDSIZE) {
                inputPlane[4][i][j] = 1;
            }
        }
    }

    if (teban == 1) {
        rep (i, BOARDSIZE) {
            rep (j, BOARDSIZE) {
                inputPlane[5][i][j] = 1;
            }
        }
    }
    else {
        rep (i, BOARDSIZE) {
            rep (j, BOARDSIZE) {
                inputPlane[5][i][j] = -1;
            }
        }
    }

    if (debugFlag & 0b1000) {
        for (auto plane : inputPlane) {
            for (auto row : plane) {
                for (auto col : row) {
                    cerr << col << " ";
                }
                cerr << endl;
            }
            cerr << endl;
        }
    }

    return inputPlane;
}



int cnt = 0;  ////////////



int ConvertChar(char s)
{
    int output = GPTALPHABET.find(s);

    if (output == -1) {
        output = GPTAlapabet.find(s);
        if (output == -1) {
            assert(false && "👺ConvertChar error");
        }
    }

    // cerr << "ConvertChar: " << s << " -> " << output + 1 << endl;  ////////////////

    return output + 1;
}


char ConvertInt(int n)
{
    assert(n >= 1 && n <= BOARDSIZE);

    char output = GPTALPHABET[n - 1];

    // cerr << "ConvertInt: " << n << " -> " << output << endl;  ////////////////

    return output;
}

double dfs(goBoard* ptr)
{
    // print("dfs", cnt);////////////////
    if (ptr->isEnded) {
        return ptr->CountResult();
    }

    tuple<char, char, char> legalMove = ptr->GenRandomMove();

    double tmp = dfs(ptr->PutStone(get<0>(legalMove), get<1>(legalMove), get<2>(legalMove)));
    // if (ptr->parent->isRoot) {
    //     for (auto x : ptr->childrens) {
    //         delete(x.second);
    //     }
    // }
    // if (!ptr->isRoot){
    //     delete ptr;
    // }
    return tmp;
}


// int MonteCarloTreeSearch()
// {
//     json j = json::parse("[[0, 0, 2, 2, 2, 1, 0, 0, 0], [0, 0, 0, 2, 1, 1, 1, 0, 0], [0, 0, 2, 2, 2, 2, 1, 1, 0], [0, 0, 0, 2, 1, 2, 1, 1, 0], [0, 2, 2, 2, 1, 2, 2, 1, 2], [0, 1, 2, 1, 1, 2, 1, 2, 0], [0, 2, 1, 1, 1, 1, 1, 0, 1], [0, 2, 2, 2, 2, 2, 1, 1, 2], [0, 0, 0, 0, 0, 2, 1, 0, 0]]");
//     vector<vector<char>> v = j;

//     goBoard* root(new goBoard(v, 1));


//     vector<tuple<char, char, char>> legalMoves = root->GenAllLegalMoves();

//     for (auto [y, x, t] : legalMoves) {
//         goBoard* tmp = root->PutStone(y, x, t);
//     }




//     for (auto x = *begin(root->ucts); get<0>(x) <= 0.0; x = *begin(root->ucts)) {
//         ++cnt;  //////////

//         if (get<0>(x) != 0.0) break;

//         double rslt = dfs(root->childrens[get<3>(x)]);

//         // print("rslt", rslt);////////

//         int win = 0;
//         if (rslt > 0) {
//             win = 1;
//         }
//         else if (rslt < 0) {
//             win = 0;
//         }

//         int numWin = get<2>(x) + win;
//         int numVisit = get<1>(x) + 1;
//         ++root->numVisits;

//         // cerr << "numWin: " << numWin << endl;
//         // cerr << "numVisit: " << numVisit << endl;
//         // cerr << "root->numVisits: " << root->numVisits << endl;

//         double uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);

//         // print("uct", uct);


//         root->ucts.erase(x);

//         root->ucts.insert(make_tuple(uct, numVisit, numWin, get<3>(x)));
//     }


//     for (; cnt < 300; ++cnt) {
//         auto x = *rbegin(root->ucts);
//         auto [uct, numWin, numVisit, move] = x;

//         goBoard* tmpp = root->PutStone(get<0>(move), get<1>(move), root->teban);

//         double rslt = dfs(tmpp);

//         delete root->childrens[move];



//         int win = 0;
//         if (rslt > 0) {
//             win = 1;
//         }
//         else if (rslt < 0) {
//             win = 0;
//         }

//         numWin += win;
//         numVisit += 1;
//         ++root->numVisits;

//         uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);
//         root->ucts.erase(x);
//         auto tmp = make_tuple(uct, numVisit, numWin, move);
//         root->ucts.insert(tmp);
//         print(tmp, rslt);
//     }


//     print("end");
//     for (auto x : root->ucts) {
//         print(x);
//     }


//     auto ans = *rbegin(root->ucts);

//     for (auto x : root->ucts) {
//         if (get<1>(x) > get<1>(ans)) {
//             ans = x;
//         }
//         else if (get<1>(x) == get<1>(ans) && get<0>(x) >= get<0>(ans)) {
//             ans = x;
//         }
//     }
//     print("ans", ans);

//     root->PrintBoard(0b1);

//     root->SucceedRoot(rootPtr, get<3>(ans));
//     root = rootPtr;


//     legalMoves = root->GenAllLegalMoves();

//     for (auto [y, x, t] : legalMoves) {
//         goBoard* tmp = root->PutStone(y, x, t);
//     }
//     for (auto x = *begin(root->ucts); get<0>(x) <= 0.0; x = *begin(root->ucts)) {
//         ++cnt;  //////////

//         if (get<0>(x) != 0.0) break;

//         double rslt = dfs(root->childrens[get<3>(x)]);

//         // print("rslt", rslt);////////

//         int win = 0;
//         if (rslt > 0) {
//             win = 1;
//         }
//         else if (rslt < 0) {
//             win = 0;
//         }

//         int numWin = get<2>(x) + win;
//         int numVisit = get<1>(x) + 1;
//         ++root->numVisits;

//         // cerr << "numWin: " << numWin << endl;
//         // cerr << "numVisit: " << numVisit << endl;
//         // cerr << "root->numVisits: " << root->numVisits << endl;

//         double uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);

//         // print("uct", uct);


//         root->ucts.erase(x);

//         root->ucts.insert(make_tuple(uct, numVisit, numWin, get<3>(x)));
//     }

//     for (; cnt < 300; ++cnt) {
//         auto x = *rbegin(root->ucts);
//         auto [uct, numWin, numVisit, move] = x;

//         goBoard* tmpp = root->PutStone(get<0>(move), get<1>(move), root->teban);

//         double rslt = dfs(tmpp);

//         delete root->childrens[move];



//         int win = 0;
//         if (rslt > 0) {
//             win = 1;
//         }
//         else if (rslt < 0) {
//             win = 0;
//         }

//         numWin += win;
//         numVisit += 1;
//         ++root->numVisits;

//         uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);
//         root->ucts.erase(x);
//         auto tmp = make_tuple(uct, numVisit, numWin, move);
//         root->ucts.insert(tmp);
//         print(tmp, rslt);
//     }


//     print("end");
//     for (auto x : root->ucts) {
//         print(x);
//     }


//     ans = *rbegin(root->ucts);

//     for (auto x : root->ucts) {
//         if (get<1>(x) > get<1>(ans)) {
//             ans = x;
//         }
//         else if (get<1>(x) == get<1>(ans) && get<0>(x) >= get<0>(ans)) {
//             ans = x;
//         }
//     }
//     print("ans", ans);

//     root->PrintBoard(0b1);

//     cerr << rootPtr << endl;
//     root->SucceedRoot(rootPtr, get<3>(ans));
//     cerr << rootPtr << endl;


//     return 0;
// }


// int PutStoneCnt = 0;





// ループを制御するためのフラグ
std::atomic<bool> running(true);

// とりあえずグローバル
int visit_Limit = 100000;

void SearchLoop(goBoard* rootPtr, TensorRTOnnxIgo& tensorRT)
{
    // (PutStone or new) -> (ExpandNode) -> (PutStone) ...
    // rootPtr new -> rootPtr.ExpandNode をしておく。
    // ptr0 -> ... -> if !ptr3->children.count(nextMove) -> PutStone -> ExpandNode -> ptr2 の ucts を更新 -> ... -> ptr0 の ucts を更新。 という流れを繰り返す。
    // ExpandNode で 合法手に対して ucts.insert される。
    // PutStone の手が連続2回目のpassのとき isEnded = true になる。


    //     auto saiki = [tensorRT](auto self, goBoard* ptr) -> tuple<int, float, float, float>
    //     {

    // #ifdef dbg_flag
    //         g_node_cnt++;
    // #endif
    //         int color = ptr->teban;

    //         if (ptr->isEnded) {
    //             double rslt = ptr->CountResult();
    //             if (rslt == 0) {
    //                 return make_tuple(color, 0.0, 1.0, 0.0);
    //             }
    //             if ((color == 1 && rslt > 0) || (color == 2 && rslt < 0)) {
    //                 return make_tuple(color, 0.0, 0.0, 1.0);
    //             }
    //             return make_tuple(color, 1.0, 0.0, 0.0);
    //         }

    //         lock_guard<mutex> lock(ptr->uctsMutex);

    //         assert(ptr->ucts.size());

    //         pair<char, char> nextMove = get<3>(*rbegin(ptr->ucts));


    //         if (!ptr->childrens.count(nextMove)) {
    //             goBoard* nextPtr = ptr->PutStone(nextMove.first, nextMove.second, color);

    //             int nextColor = nextPtr->teban;

    //             if (nextPtr->isEnded) {
    // #ifdef dbg_flag
    //                 ++endCnt;
    // #endif
    //                 double rslt = nextPtr->CountResult();
    //                 if (rslt == 0) {
    //                     return make_tuple(nextColor, 0.0, 1.0, 0.0);
    //                 }
    //                 /// TODO: 正しいか確認
    //                 if ((nextColor == 1 && rslt > 0) || (nextColor == 2 && rslt < 0)) {
    //                     return make_tuple(nextColor, 0.0, 0.0, 1.0);
    //                 }
    //                 return make_tuple(nextColor, 1.0, 0.0, 0.0);
    //             }

    //             return nextPtr->ExpandNode(tensorRT);
    //         }


    //         tuple<int, float, float, float> returnData = self(self, ptr->childrens[nextMove]);

    //         ptr->UpdateUcts(returnData, nextMove);

    //         return returnData;
    //     };


    //     while (running.load()) {
    //         if (rootPtr->isEnded) {
    //             break;
    //         }
    //         if (rootPtr->numVisits > visitMax || rootPtr->numVisits > visit_Limit) {
    //             sleep(0.1);
    //         }
    //         saiki(saiki, rootPtr);
    //     }

    //     return;


    // 再帰なしバージョン。leaf（ExpandNode で作ったノード・isEndedのノード）ではuctの更新はしない。
    while (running.load()) {
        if (rootPtr->isEnded) {
            break;
        }
        if (rootPtr->numVisits > visitMax || rootPtr->numVisits > visit_Limit) {
            sleep(0.1);
        }

        // 潜る探索ループ
        goBoard* ptr = rootPtr;
        tuple<int, float, float, float> leafRslt;
        while (true) {
            char color = ptr->teban;

            // もし終局（直前2手がパス）なら結果を leafRslt に入れてbreak
            /// TODO: これ通ったら
            if (ptr->isEnded) {
                double tmpRslt = ptr->CountResult();
                if (tmpRslt == 0) {
                    leafRslt = make_tuple(color, 0.0, 1.0, 0.0);
                    break;
                }
                if ((color == 1 && tmpRslt > 0) || (color == 2 && tmpRslt < 0)) {
                    leafRslt = make_tuple(color, 0.0, 0.0, 1.0);
                    break;
                }
                leafRslt = make_tuple(color, 1.0, 0.0, 0.0);
                break;
            }

            lock_guard<mutex> lock(ptr->uctsMutex);

            assert(ptr->ucts.size());

            // uct が最大の手を取得
            pair<char, char> nextMove = get<3>(*rbegin(ptr->ucts));

            // その手が未展開なら展開して結果を leafRslt に入れてbreak
            if (!ptr->childrens.count(nextMove)) {
                goBoard* nextPtr = ptr->PutStone(nextMove.first, nextMove.second, color);

                int nextColor = nextPtr->teban;

                /// TODO: ここで終局判定する必要ある？
                if (nextPtr->isEnded) {
#ifdef dbg_flag
                    ++endCnt;
#endif
                    double tmpRslt = nextPtr->CountResult();
                    if (tmpRslt == 0) {
                        leafRslt = make_tuple(nextColor, 0.0, 1.0, 0.0);
                    }
                    /// TODO: 正しいか確認
                    else if ((nextColor == 1 && tmpRslt > 0) || (nextColor == 2 && tmpRslt < 0)) {
                        leafRslt = make_tuple(nextColor, 0.0, 0.0, 1.0);
                    }
                    else {
                        leafRslt = make_tuple(nextColor, 1.0, 0.0, 0.0);
                    }
                    break;
                }

                leafRslt = nextPtr->ExpandNode(tensorRT);
#ifdef dbg_flag
                ++expandCnt;
#endif
                break;
            }


            ptr = ptr->childrens[nextMove];
        }

        // 浮かんでいく探索ループ
        pair<char, char> nextMove = ptr->previousMove;
        while (true) {
            ptr->UpdateUcts(leafRslt, nextMove);

            if (ptr->isRoot) {
                break;
            }

            nextMove = ptr->previousMove;
            ptr = ptr->parent;
        }
    }

    return;
}

string Gpt(const string input, goBoard*& rootPtr, TensorRTOnnxIgo& tensorRT, thread& searchThread, int thinkTime = 5, bool ponder = true)
{
    // cerr << "Gpt input: " << input << endl;  /////////////////////

    stringstream ss{input};
    string s;
    vector<string> commands;

    string output;
    // スペース（' '）で区切って，vに格納
    while (getline(ss, s, ' ')) {
        commands.push_back(s);
    }


    if (commands[0] == "list_commands") {
        output = "=list_commands\nname\nboardsize\nclear_board\nkomi\nplay\ngenmove\nquit\nshowboard\n";
    }
    else if (commands[0] == "name") {
        output = "=TantamaGo";
    }
    else if (commands[0] == "protocol_version") {
        output = "=2";
    }
    else if (commands[0] == "version") {
        output = "=0.1";
    }
    else if (commands[0] == "boardsize") {
        if (stoi(commands[1]) != BOARDSIZE) {
            output = "dismatch_boardsize";
            goto GOTO_GPT_SEND;
        }
        output = "=";
    }
    else if (commands[0] == "clear_board") {
        running.store(false);
        if (searchThread.joinable()) {
            searchThread.join();
        }

        delete rootPtr;
        rootPtr = nullptr;
        rootPtr = new goBoard();
        rootPtr->ExpandNode(tensorRT);

        if (ponder) {
            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
        }

        output = "=";
    }
    else if (commands[0] == "komi") {
        output = "=";
    }
    else if (commands[0] == "play") {
        if (rootPtr->isEnded) {
            if (commands[0] != "clear_board") {
                output = "game has already ended";
                goto GOTO_GPT_SEND;
            }
        }

        if (commands.size() != 3) {
            output = "unknown_command";
        }
        else {
            char y, x;

            // print(commands[1]);  /////////////////////
            // print(commands[2]);  /////////////////////

            if (commands[2] == "pass" || commands[2] == "PASS" || commands[2] == "pass\n" || commands[2] == "PASS\n") {
                y = x = 0;
            }
            else {
                x = ConvertChar(commands[2][0]);
                if (x == -1) {
                    output = "dismatch_boardsize";
                    cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    print("x, y: ", x, y);                                   //////////////
                    goto GOTO_GPT_SEND;
                }

                if (commands[2].size() == 2) {
                    y = commands[2][1] - '0';
                }
                else if (commands[2].size() == 3) {
                    if (commands[2][2] == '\n') {
                        y = commands[2][1] - '0';
                    }
                    else {
                        y = (commands[2][1] - '0') * 10 + commands[2][2] - '0';
                    }
                }
                else {
                    output = "unknown_command";
                    cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    goto GOTO_GPT_SEND;
                }


                if (y < 1 || y > BOARDSIZE) {
                    output = "dismatch_boardsize";
                    cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    print("x, y: ", x, y);                                   //////////////
                    goto GOTO_GPT_SEND;
                }

                if (commands[1] == "black" || commands[1] == "b" || commands[1] == "B") {
                    if (rootPtr->teban != 1) {
                        cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                        output = "dismatch_color";
                        goto GOTO_GPT_SEND;
                    }
                }
                else if (commands[1] == "white" || commands[1] == "w" || commands[1] == "W") {
                    if (rootPtr->teban != 2) {
                        cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                        output = "dismatch_color";
                        goto GOTO_GPT_SEND;
                    }
                }
                else {
                    output = "unknown_command";
                    cout << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    goto GOTO_GPT_SEND;
                }
            }

            running.store(false);
            if (searchThread.joinable()) {
                searchThread.join();
            }

            if (debugFlag & 1 << 5) {
                cout << "\n--------------------\n"
                     << "rootPtr: " << rootPtr << ", teban: " << int(rootPtr->teban) << ", moveCnt: " << rootPtr->moveCnt << endl;  //////////////////////////
#ifdef dbg_flag
                cout << "expandCnt: " << expandCnt << ", endCnt: " << endCnt << endl;
#endif
                print();
                // rootPtr->PrintBoard(1 << 29);
                // print();
                // rootPtr->PrintBoard(1 << 26);  //////////////////
                // print();
                // rootPtr->PrintBoard(1 << 28);  //////////////////
                // print();
                // rootPtr->PrintBoard(1 << 31);
                // print();
                rootPtr->PrintBoard(1 << 27);
                print();
                rootPtr->PrintBoard(0b1);
                print();
            }

            if (rootPtr->childrens.size() == 0) {
                rootPtr->ExpandNode(tensorRT);
            }
            rootPtr = rootPtr->SucceedRoot(rootPtr, {y, x});
            if (rootPtr->childrens.size() == 0 && !rootPtr->isEnded) {
                rootPtr->ExpandNode(tensorRT);
            }
            if (rootPtr->isEnded) {
                goto GOTO_GPT_SEND;
            }

            if (ponder) {
                running.store(true);
                searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
            }

            output = "=";
        }
    }
    else if (commands[0] == "genmove") {
        if (rootPtr->isEnded) {
            output = "game has already ended";
            goto GOTO_GPT_SEND;
        }


        if (!ponder) {
            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
        }

        sleep(thinkTime);  ///////////////

        if (debugFlag & 1 << 5) {
            cout << "\n--------------------\n"
                 << "rootPtr: " << rootPtr << ", teban: " << int(rootPtr->teban) << ", moveCnt: " << rootPtr->moveCnt << endl;  //////////////////////////
#ifdef dbg_flag
            cout << "expandCnt: " << expandCnt << ", endCnt: " << endCnt << endl;
#endif
            print();
            rootPtr->PrintBoard(0b100);
            print();
            rootPtr->PrintBoard(1 << 29);
            print();
            // rootPtr->PrintBoard(1 << 26);  //////////////////
            // print();
            rootPtr->PrintBoard(1 << 28);  //////////////////
            print();
            rootPtr->PrintBoard(1 << 31);
            print();
            rootPtr->PrintBoard(1 << 27);
            print();
            rootPtr->PrintBoard(0b1);
            print();
        }


        running.store(false);
        if (searchThread.joinable()) {
            searchThread.join();
        }


        pair<char, char> move = rootPtr->GetBestMove();
        // cerr << "move: " << (int)move.first << " " << (int)move.second << endl;////////////////

        if (move.first == 0 && move.second == 0) {
            output = "=pass";
        }
        else {
            output = "=";
            output += ConvertInt(move.second);
            output += to_string(move.first);
        }

        if (rootPtr->childrens.size() == 0) {
            rootPtr->ExpandNode(tensorRT);
        }
        rootPtr = rootPtr->SucceedRoot(rootPtr, move);
        if (rootPtr->childrens.size() == 0 && !rootPtr->isEnded) {
            rootPtr->ExpandNode(tensorRT);
        }
        if (rootPtr->isEnded) {  ////////////????
            goto GOTO_GPT_SEND;
        }

        if (ponder) {
            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
        }
    }
    else if (commands[0] == "quit") {
        running.store(false);
        if (searchThread.joinable()) {
            searchThread.join();
        }

        output = "quit";
        goto GOTO_GPT_SEND;
    }
    else if (commands[0] == "showboard") {
        running.store(false);
        if (searchThread.joinable()) {
            searchThread.join();
        }

        output = "=";
        rootPtr->PrintBoard(0b1);

        if (ponder) {
            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
        }
    }
    else if (commands[0] == "_print") {
        rootPtr->PrintBoard(1 << 28);
        print();
        rootPtr->PrintBoard(1 << 27);
        print();
        rootPtr->PrintBoard(1 << 31);
        print();
        rootPtr->PrintBoard(1 << 30);
        print();
        rootPtr->PrintBoard(1 << 29);
        print();
        rootPtr->PrintBoard(0b1);
        print();
        output = "=";
    }
    else {
        output = "unknown_command";
    }

GOTO_GPT_SEND:;

    // cerr << "Gpt output: " << output << endl;  ////////////////
    // // cerr << "Gpt: " << input << " -> " << output << endl;////////////////

    return output;
}


int suiron(int n)
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();

    print("build end");  ////////////////////


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(tensorRT);


    int saikiCnt = 0;

    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));

    sleep(n);
    running.store(false);
    if (searchThread.joinable()) {
        searchThread.join();
    }


    goBoard* tmp = rootPtr;
    while (true) {
        tmp->PrintBoard(1 << 28);
        print();
        tmp->PrintBoard(1 << 27);
        print();
        tmp->PrintBoard(1 << 31);
        print();
        tmp->PrintBoard(1 << 30);
        print();
        tmp->PrintBoard(1 << 29);
        print();
        tmp->PrintBoard(0b1);
        print();

        int y = 123, x = 123;
        while (!tmp->childrens.count({y, x})) {
            cerr << "input y: ";
            cin >> y;
            if (y == -1) {
                tmp = tmp->parent;
                goto PASS;
            }
            cerr << "input x: ";
            cin >> x;
            if (x == -1) {
                goto END;
            }
        }
        tmp = tmp->childrens[{y, x}];
    PASS:;
    }

END:;


    cerr << "scceed" << endl;
    int x, y;
    cerr << "input y: ";
    cin >> y;
    cerr << "input x: ";
    cin >> x;

    if (rootPtr->childrens.size() == 0) {
        rootPtr->ExpandNode(tensorRT);
    }
    rootPtr = rootPtr->SucceedRoot(rootPtr, {y, x});
    if (rootPtr->childrens.size() == 0 && !rootPtr->isEnded) {
        rootPtr->ExpandNode(tensorRT);
    }

    tmp = rootPtr;
    while (true) {
        tmp->PrintBoard(1 << 28);
        print();
        tmp->PrintBoard(1 << 27);
        print();
        tmp->PrintBoard(1 << 31);
        print();
        tmp->PrintBoard(1 << 30);
        print();
        tmp->PrintBoard(1 << 29);
        print();
        tmp->PrintBoard(0b1);
        print();

        int y = 123, x = 123;
        while (!tmp->childrens.count({y, x})) {
            cerr << "input y: ";
            cin >> y;
            if (y == -1) {
                tmp = tmp->parent;
                goto PASS2;
            }
            cerr << "input x: ";
            cin >> x;
            if (x == -1) {
                goto END2;
            }
        }
        tmp = tmp->childrens[{y, x}];
    PASS2:;
    }

END2:;

    return 0;
}





int PlayWithGpt()
{
    int thinkTime = 10;
    // cout << "thinkTime << ";
    // cin >> thinkTime;
    int visit_Limit = 100000;
    // cout << "visit_Limit << ";
    // cin >> visit_Limit;


    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(tensorRT);

    int saikiCnt = 0;

    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));

    string input;
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        output = Gpt(input, rootPtr, tensorRT, searchThread, thinkTime, false);
        cout << output << endl;
        if (output == "exit") {
            break;
        }
    }

    return 0;
}



int GptSoket()
{
    int thinkTime = 10;
    cout << "thinkTime << ";
    cin >> thinkTime;
    cout << "visit_Limit << ";
    cin >> visit_Limit;
    int port = 8000;
    cout << "port << ";
    cin >> port;


    samplesCommon::Args args;

    // args.runInInt8 = false;
    // args.runInFp16 = true;
    // args.runInBf16 = true;
    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();

    print("build end");  ////////////////////


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(tensorRT);

    int saikiCnt = 0;

    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));

    print("thread start");  ////////////////////


    // ソケット通信
    int sockfd, client_sockfd;
    struct sockaddr_in address;
    int addrlen = sizeof(address);

    char buf[1024] = {0};

    // ソケットの作成
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    // アドレスの準備
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // ソケットにアドレスを割り当て
    if (bind(sockfd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }

    // ソケットをリッスン状態にする
    if (listen(sockfd, 3) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }

    // 新しい接続の受け入れ
    if ((client_sockfd = accept(sockfd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }

    print("server start");  ////////////////////


    // 受信
    string input = "";
    string output = "";
    int rsize;
    while (true) {
        rsize = recv(client_sockfd, buf, sizeof(buf), 0);

        input = buf;

        if (rsize == 0) {
            break;
        }


        cerr << "recv data: " << buf << endl;  /////////////////////

        output = Gpt(buf, rootPtr, tensorRT, searchThread, thinkTime, false);


        // if (input.substr(0, 4) == "genm" || input.substr(0, 4) == "play") {
        //     cout << "\n--------------------\n"
        //          << "rootPtr: " << rootPtr << endl;  //////////////////////////
        //     print();
        //     rootPtr->PrintBoard(1 << 26);  //////////////////
        //     print();
        //     rootPtr->PrintBoard(1 << 28);  //////////////////
        //     print();
        //     rootPtr->PrintBoard(1 << 31);
        //     print();
        //     rootPtr->PrintBoard(1 << 27);
        //     print();
        //     rootPtr->PrintBoard(1 << 29);
        //     print();
        //     rootPtr->PrintBoard(0b1);
        //     print();
        // }


        if (output == "quit") {
            output = "=";
            cerr << "send data: " << output << endl;  /////////////////////
            output += "\n";
            write(client_sockfd, output.c_str(), output.length());
            break;
        }

        cerr << "send data: " << output << endl;  /////////////////////

        output += "\n";
        write(client_sockfd, output.c_str(), output.length());

        // Clear the buffer after sending the data
        memset(buf, 0, sizeof(buf));

        sleep(0.1);
    }


    // ソケットクローズ
    close(client_sockfd);
    close(sockfd);

    running.store(false);
    if (searchThread.joinable()) {
        searchThread.join();
    }

    return 0;
}


int Test()
{
    string s;
    int teban;
    cerr << "input teban << ";
    getline(cin, s);
    teban = stoi(s);
    cerr << "input s << ";
    getline(cin, s);
    print("teban: ", teban);  ////////////////////
    print("s: ", s);          ////////////////////
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    json j = json::parse(s);
    vector<vector<char>> v = j;

    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard(v, teban);

    rootPtr->PrintBoard(0b1);
    rootPtr->ExpandNode(tensorRT);
    int saikiCnt = 0;

    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));


    string input = "";
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        output = Gpt(input, rootPtr, tensorRT, searchThread);
        cout << output << endl;
        if (output == "quit") {
            break;
        }
    }

    return 0;
}



int main(int argc, char* argv[])
{
    // MonteCarloTreeSearch();

    // int n = 10;
    // if (argc == 2) n = stoi(argv[1]);
    // suiron(n);

    // Test();

    // PlayWithGpt();

    GptSoket();

    return 0;
}
