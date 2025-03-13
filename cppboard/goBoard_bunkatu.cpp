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

// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ python3 bin/cgosclient.py simple.cfg




#ifndef myMacro_hpp_INCLUDED
#include "../myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif



#ifndef config_hpp_INCLUDED
#include "config_bunkatu.hpp"
#define config_hpp_INCLUDED
#endif



// #ifndef tensorRTigo_cpp_INCLUDED
// #include "./tensorRTigo.cpp"
// #define tensorRTigo_cpp_INCLUDED
// #endif

#ifndef goBoard_hpp_INCLUDED
#include "goBoard_bunkatu.hpp"
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


tuple<int, float, float, float> goBoard::ExpandNode(pair<vector<float>, vector<float>> input)
{
    assert(this->ucts.size() == 0);
    assert(input.first.size() == BOARDSIZE * BOARDSIZE + 1);
    assert(input.second.size() == 3 || !(cerr << "input.second.size(): " << input.second.size() << endl));
    // assert(!isEnded);

    // this->isNotExpanded = false;

    vector<float> tmpPolicys = input.first;

    this->values = input.second;

    vector<tuple<char, char, char>> legalMoves = GenAllLegalMoves();

    numVisits = legalMoves.size();



    for (auto [y, x, t] : legalMoves) {
        if (x == 0 && y == 0) {
            float tmp = tmpPolicys[BOARDSIZE * BOARDSIZE];
            policys[make_pair(y, x)] = tmp;
            continue;
        }
        float tmp = tmpPolicys[(y - 1) * BOARDSIZE + x - 1];
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
    else {  // softmaxあり
        // softmaxで使う変数
        float maxPolicy = 0.0;

        for (auto [y, x, t] : legalMoves) {
            if (x == 0 && y == 0) {
                float tmp = tmpPolicys[BOARDSIZE * BOARDSIZE];
                policys[make_pair(y, x)] = tmp;
                chmax(maxPolicy, tmp);
                continue;
            }
            float tmp = tmpPolicys[(y - 1) * BOARDSIZE + x - 1];
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
        // tmppolicyにsoftmaxここまで


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
        cerr << "policys勝率*1000 policys.size():" << policys.size() << endl;
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

        cerr << "探索後勝率*1000 の表示。" << endl << "ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
#ifdef dbg_flag
        cerr << "endCnt: " << endCnt << ", depth: " << deepestMoveCnt - this->moveCnt << endl;
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

    // // countResultの表示
    // if (bit & 1 << 24) {
    //     CountResult(true);
    // }


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

int goBoard::CountLiberties(int y, int x, vector<vector<char>> board = {})
{
    assert(x >= 0 && x <= BOARDSIZE + 1 && y >= 0 && y <= BOARDSIZE + 1);

    if (board.empty()) {
        board = this->board;
    }

    if (board[y][x] == 0) {
        return -1;
    }

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

bool goBoard::IsBestMoveCrucial()
{
    /// TODO: ロックする場所やタイミングの見直し。
    lock_guard<mutex> lock(uctsMutex);

    assert(this->ucts.size());

    // expand -> uct最大の手をputstone -> 置いた石周りのlib数を数える -> lib数が1から増える、または1になる場合は -> crucialならすぐにexpand の方がいい？
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

    /// TODO:無い方がいい気がする
    bool isFillEye = true;
    int tmpId = -2;
    for (auto dir : directions) {
        int nx = x + (int)dir.first;
        int ny = y + (int)dir.second;

        if (board[ny][nx] == 0) {
            isFillEye = false;
            break;
        }

        if (board[ny][nx] == 3) {
            continue;
        }

        if (tmpId == -2) {
            tmpId = idBoard[ny][nx];
            continue;
        }

        if (tmpId != idBoard[ny][nx]) {
            isFillEye = false;
            break;
        }
    }

    if (isFillEye) {
        return 4;
    }

    return 0;
};


vector<tuple<char, char, char>> goBoard::GenAllLegalMoves()
{
    // assert(!isEnded);

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

double goBoard::CountResult(bool dbg = false)
{
    /// TODO: 日本ルール用の暫定措置。どうにかしたい。白黒が隣り合っているところでラインを引いて地を数える？中国ルールで最後までプレイしてみる？
    // if (1) {  ////////////////////
    if (isJapaneseRule) {
        assert(this->ucts.size());
        assert(this->values.size());

        double tmpScore;
        if (values.size()) {
            if (values[0] > values[2] && values[0] > values[1]) {
                tmpScore = -1.0;
            }
            else if (values[1] > values[0] && values[1] > values[2]) {
                tmpScore = 0.0;
            }
            else {
                tmpScore = 1.0;
            }

            if (teban == 1) {
                return tmpScore;
            }
            else {
                return tmpScore * -1;
            }
        }
        else if (parent->values.size()) {
            if (parent->values[0] > parent->values[2] && parent->values[0] > parent->values[1]) {
                tmpScore = -1.0;
            }
            else if (parent->values[1] > parent->values[0] && parent->values[1] > parent->values[2]) {
                tmpScore = 0.0;
            }
            else {
                tmpScore = 1.0;
            }

            if (teban == 1) {
                return tmpScore;
            }
            else {
                return tmpScore * -1;
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

    // this->PrintBoard(0b1);

    auto saiki1 = [](auto self, vector<vector<char>>& tmpChecked, char y, char x) -> int
    {
        if (tmpChecked[y][x] == 0) {
            tmpChecked[y][x] = 1;

            char tmpColor;
            for (auto next : directions) {
                char nx = x + next.first;
                char ny = y + next.second;

                if (self(self, tmpChecked, ny, nx)) {
                    return 1;
                }
            }
        }
        else if (tmpChecked[y][x] == 2) {
            return 1;
        }

        return 0;
    };

    auto saiki2 = [](auto self, vector<vector<char>>& tmpChecked, char y, char x) -> int
    {
        if (tmpChecked[y][x] == 0) {
            tmpChecked[y][x] = 2;

            char tmpColor;
            for (auto next : directions) {
                char nx = x + next.first;
                char ny = y + next.second;

                if (self(self, tmpChecked, ny, nx)) {
                    return 1;
                }
            }
        }
        else if (tmpChecked[y][x] == 1) {
            return 1;
        }

        return 0;
    };

    vector<vector<char>> tmpBoard = this->board;
    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (tmpBoard[i][j] == 0) {
                vector<vector<char>> tmpChecked = tmpBoard;
                if (!saiki1(saiki1, tmpChecked, i, j)) {
                    tmpBoard = tmpChecked;
                }
                else {
                    tmpChecked = tmpBoard;
                    if (!saiki2(saiki2, tmpChecked, i, j)) {
                        tmpBoard = tmpChecked;
                    }
                }
            }
            if (tmpBoard[i][j] == 1) {
                ++blackScore;
            }
            else if (tmpBoard[i][j] == 2) {
                ++whiteScore;
            }
        }
    }

    if (dbg) {//////////////////////////
        cerr << "blackScore: " << blackScore << ", whiteScore: " << whiteScore << ", score: " << blackScore - whiteScore - komi << endl;  ///////////////

        // print tmpBoard
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << (int)tmpBoard[i][j] << " " << flush;
            }
            cerr << endl;
        }
    }

    return blackScore - whiteScore - komi;


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