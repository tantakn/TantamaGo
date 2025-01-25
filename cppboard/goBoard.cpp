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


// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ g++ -Wall -Wno-deprecated-declarations -std=c++17   -I"./TensorRT/common"   -I"./TensorRT/utils"   -I"./TensorRT"   -I"/usr/local/cuda/include"   -I"./TensorRT/include"   -D_REENTRANT -DTRT_STATIC=0   -g  goBoard.cpp   ./TensorRT/common/bfloat16.cpp   ./TensorRT/common/getOptions.cpp   ./TensorRT/common/logger.cpp   ./TensorRT/common/sampleDevice.cpp   ./TensorRT/common/sampleEngines.cpp   ./TensorRT/common/sampleInference.cpp   ./TensorRT/common/sampleOptions.cpp   ./TensorRT/common/sampleReporting.cpp   ./TensorRT/common/sampleUtils.cpp   ./TensorRT/utils/fileLock.cpp   ./TensorRT/utils/timingCache.cpp   -o goBoard   -L"/usr/local/cuda/lib64"   -Wl,-rpath-link="/usr/local/cuda/lib64"   -L"./TensorRT/lib"   -Wl,-rpath-link="./TensorRT/lib"   -L"./TensorRT/bin"   -Wl,--start-group   -lnvinfer   -lnvinfer_plugin   -lnvonnxparser   -lcudart   -lrt   -ldl   -lpthread   -Wl,--end-group   -Wl,--no-relax




#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif


constexpr int BOARDSIZE = 19;
// constexpr int BOARDSIZE = 9;

constexpr double komi = 7.5;

constexpr ll debugFlag = 0;
// constexpr ll debugFlag = ll(1)<<25;
// constexpr ll debugFlag = ll(1)<<31 | ll(1)<<29 | ll(1)<<30;

const string tensorRTModelPath = "./19ro.onnx";
// const string tensorRTModelPath = "./test19_2.onnx";


#ifndef tensorRTigo_cpp_INCLUDED
#include "./tensorRTigo.cpp"
#define tensorRTigo_cpp_INCLUDED
#endif

#ifndef goBoard_hpp_INCLUDED
#include "goBoard.hpp"
#define goBoard_hpp_INCLUDED
#endif


#define dbg_flag
#ifdef dbg_flag
ll g_node_cnt = 0;
#endif


mt19937 mt(random_device{}());

/// @brief ルートノード。グローバル変数。
goBoard* rootPtr = nullptr;


goBoard::goBoard()
    : board(rawBoard), idBoard(rawIdBoard), teban(1), parent(nullptr), isRoot(true), moveCnt(0)
{
    assert(rootPtr == nullptr);
    rootPtr = this;

    libs[-1] = INF;
}

goBoard::goBoard(vector<vector<char>> inputBoard, char inputTeban)
    : board(InputBoardFromVec(inputBoard)), idBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0)), teban(inputTeban), isRoot(true), moveCnt(0)
{
    /// TODO: teban の扱いを考える

    assert(rootPtr == nullptr);
    rootPtr = this;

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

#ifdef dbg_flag
    g_node_cnt++;
#endif

    if (x == 0 && y == 0) {
        if (parent->previousMove == make_pair((char)0, (char)0)) {
            isEnded = true;
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

goBoard* goBoard::SucceedRoot(goBoard* rootPtr, pair<char, char> move)
{
    /// TODO: 最初の root を宣言したポインタが rootptr 以外だとここからは変えられないから、安全ではないのでは？

    assert(isRoot);
    assert(rootPtr == this);
    assert(childrens.count(move));

    if (!childrens.count(move)) {
        PutStone(move.first, move.second, teban);
    }


    goBoard* tmp = childrens[move];
    childrens.erase(move);
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


    ++numVisits;

    vector<tuple<char, char, char>> legalMoves = GenAllLegalMoves();

    /// 推論の結果を一時保存する配列。tmpPolicy[BOARDSIZE * BOARDSIZE] はパス。
    vector<float> tmpPolicy(BOARDSIZE * BOARDSIZE + 1, 0.0);


    tensorRT.infer(MakeInputPlane(), tmpPolicy, values);
    if (debugFlag & ll(1) << 25) {
        print("tmpPolicy.size():", tmpPolicy.size());  //////////////
        rep (i, BOARDSIZE) {
            rep (j, BOARDSIZE) {
                cerr << showpos << fixed << setprecision(4) << tmpPolicy[i * BOARDSIZE + j] << " ";
            }
            cerr << endl;
        }
        cerr << fixed << setprecision(4) << tmpPolicy[BOARDSIZE * BOARDSIZE] << endl;
        print("values.size():", values.size());
        print("values:", values);                     ////////////////
        cerr << resetiosflags(std::ios::floatfield);  // 浮動小数点の書式をリセット
        cerr << resetiosflags(std::ios::showpoint);   // showpoint をリセット
        cerr << resetiosflags(std::ios::showpos);     // showpos をリセット
        cerr << std::defaultfloat;
    }


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

    // policyにsoftmax
    bunbo = 0.0;
    for (auto [move, x] : policys) {
        bunbo += exp(x - maxPolicy);
    }

    float maxPolicy2 = 0.0;
    for (auto [move, x] : policys) {
        policys[move] = exp(x - maxPolicy) / bunbo;
        chmax(maxPolicy2, policys[move]);
    }


    // policys の最大が values[2] になるように調整
    for (auto [move, x] : policys) {
        policys[move] = values[2] * x / maxPolicy2;
    }


    for (auto [move, x] : policys) {
        float tmpUct = x + sqrt(2 * log(policys.size()));
        ucts.insert(make_tuple(tmpUct, 1, x, move));
    }

    if (debugFlag & 1 << 31) {
        PrintBoard(1 << 31);
    }


    return tie(teban, values[0], values[1], values[2]);
}


bool goBoard::UpdateUcts(tuple<int, float, float, float> input, pair<char, char> inputMove)
{
    auto [inputColor, inputWin, inputDraw, inputLose] = input;

    // print("inputColor:", inputColor);  ////////////
    // print("inputWin:", inputWin);
    // print("inputDraw:", inputDraw);
    // print("inputLose:", inputLose);
    // print("teban:", teban);

    /// TODO: npz作るときに逆になってることがある？多分、== が正しい？
    if (inputColor == teban) {
        swap(inputWin, inputLose);
    }

    set<tuple<double, int, float, pair<char, char>>> tmpUcts;

    ++numVisits;

    for (auto [uct, cnt, winSum, uctMove] : ucts) {
        if (inputMove == uctMove) {
            tmpUcts.insert(make_tuple((winSum + inputWin) / (cnt + 1) + sqrt(2 * log(numVisits) / (cnt + 1)),
                                      cnt + 1,
                                      winSum + inputWin,
                                      uctMove));
            // tmpUcts.insert(make_tuple(inputWin + sqrt(2 * log(numVisits) / cnt + 1), cnt + 1, winSum + inputWin, uctMove));
            continue;
        }

        tmpUcts.insert(make_tuple(winSum / cnt + sqrt(2 * log(numVisits) / cnt), cnt, winSum, uctMove));
    }

    ucts = tmpUcts;

    return 0;
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
        cerr << (int)previousMove.first << " " << (int)previousMove.second << " " << (int)teban << endl;  ////////////////
        cerr << "moveCnt: " << (int)moveCnt << ", teban: " << (int)teban << endl;
        cerr << "   1 2 3 4 5 6 7 8 9" << endl;
        rep (i, board.size()) {
            if (i == 0 || i == BOARDSIZE + 1) {
                continue;
            }
            cerr << setw(2) << setfill(' ') << i << " ";
            rep (j, board.size()) {
                if (board[i][j] == 3) {
                }
                else if (board[i][j] == 1) {
                    cerr << "● ";
                }
                else if (board[i][j] == 2) {
                    cerr << "○ ";
                }
                else if (i == 1 && j == 1) {
                    cerr << "┌ ";
                }
                else if (i == BOARDSIZE && j == 1) {
                    cerr << "└ ";
                }
                else if (i == 1 && j == BOARDSIZE) {
                    cerr << "┐ ";
                }
                else if (i == BOARDSIZE && j == BOARDSIZE) {
                    cerr << "┘ ";
                }
                else if (i == 1) {
                    cerr << "┬ ";
                }
                else if (i == BOARDSIZE) {
                    cerr << "┴ ";
                }
                else if (j == 1) {
                    cerr << "├ ";
                }
                else if (j == BOARDSIZE) {
                    cerr << "┤ ";
                }
                else {
                    cerr << "+ ";
                }
            }
            cerr << endl;
        }
    }

    if (bit & 0b010) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cerr << (int)board[i][j] << " ";
            }
            cerr << endl;
        }
        cerr << endl;
    }

    if (bit & 0b100) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cerr << setw(3) << setfill(' ') << idBoard[i][j] << " ";
            }
            cerr << endl;
        }
        cerr << endl;
    }

    if (bit & 0b1000) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(3) << setfill(' ') << libs[idBoard[i][j]] << " ";
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
                cerr << (int)board[i][j] << " ";
            }
            cerr << endl;
        }
    }

    // 推論の結果。softmax後。
    if (bit & 1 << 31) {
        print("policys.size():", policys.size());
        vector<vector<float>> tmp(BOARDSIZE + 2, vector<float>(BOARDSIZE + 2, 1.11111));
        for (auto [move, x] : policys) {
            // print(move, x);
            tmp[move.first][move.second] = x;
        }
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (board[i][j] == 0) {
                    cerr << setw(4) << setfill(' ') << static_cast<int>((tmp[i][j] - floor(tmp[i][j])) * 1000) << " ";
                    // cerr << fixed << setprecision(4) << tmp[i][j] << " ";
                }
                else if (board[i][j] == 1) {
                    cerr << "#### ";
                }
                else if (board[i][j] == 2) {
                    cerr << "@@@@ ";
                }
                else {
                    cerr << "???? ";
                }
                // cerr << fixed << setprecision(4) << tmp[i][j] << " ";
            }
            cerr << endl;
        }
        print("pass:", tmp[0][0]);  //////////////
        print("values:", values);
        cerr << resetiosflags(ios_base::floatfield);
    }

    // childの有無
    if (bit & 1 << 30) {
        print("childrens.size():", childrens.size());
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (childrens.count(make_pair(i, j))) {
                    cerr << "O ";
                }
                else {
                    cerr << "X ";
                }
            }
            cerr << endl;
        }
    }

    // uctの表示
    if (bit & 1 << 29) {
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, 0));
        int maxCnt = 0;
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = uct;
            if (cnt > maxCnt) {
                maxCnt = cnt;
                maxMove = move;
            }
        }
        cerr << "ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        cerr << "以下の表は uct値 * 10 を表示" << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << fixed << setprecision(1) << showpoint << tmp[i][j] * 10 << " ";
            }
            cerr << endl;
        }
        cerr << "pass: " << tmp[0][0] << endl;
        cerr << "ans: [" << int(maxMove.first) << ", " << int(maxMove.second) << "]" << endl;
    }

    // visitの表示
    if (bit & 1 << 28) {
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, 0));
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = cnt;
        }
        cerr << "visitの表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(4) << setfill(' ')  << int(tmp[i][j]) << " ";
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ')  << tmp[0][0] << endl;
    }

    // 勝率の表示
    if (bit & 1 << 27) {
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, 0));
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = winSum / cnt;
        }
        cerr << "勝率の表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 100) << " ";
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ')  << int(tmp[0][0] * 100) << endl;
    }

    // ペナルティの表示
    if (bit & 1 << 27) {
        vector<vector<double>> tmp(BOARDSIZE + 2, vector<double>(BOARDSIZE + 2, 0));
        pair<char, char> maxMove;
        for (auto [uct, cnt, winSum, move] : ucts) {
            tmp[move.first][move.second] = sqrt(2 * log(numVisits) / (cnt));
        }
        cerr << "ペナルティの表示。ucts.size(): " << ucts.size() << ", visit: " << numVisits << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cerr << setw(4) << setfill(' ') << int(tmp[i][j] * 100) << " ";
            }
            cerr << endl;
        }
        cerr << "pass: " << setw(4) << setfill(' ')  << int(tmp[0][0] * 100) << endl;
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
    bool isFillEye = true;
    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (board[ny][nx] == 0 || board[ny][nx] == 3 - color || libs[idBoard[ny][nx]] == 1) {
            isFillEye = false;
            /// TODO: break でいい？
        }
    }
    if (isFillEye) {
        return 4;
    }

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


    if (debugFlag & 1 << 30) {
        print("legalMoves.size():", legalMoves.size());

        cerr << "pass: ";
        if (count(legalMoves.begin(), legalMoves.end(), make_tuple(0, 0, 0))) {
            cerr << "O" << endl;
        }
        else {
            cerr << "X" << endl;
        }

        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (count(legalMoves.begin(), legalMoves.end(), make_tuple(i, j, this->teban))) {
                    cerr << "O ";
                }
                else {
                    cerr << "_ ";
                }
            }
            cerr << endl;
        }
    }


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
    if (1) {
        if (values.size()) {
            if (teban == 2) {
                return values[0];
            }
            else {
                return values[2];
            }
        }
        else if (parent->values.size()) {
            if (parent->teban == 2) {
                return parent->values[0];
            }
            else {
                return parent->values[2];
            }
        }
        else {
            assert(false && "👺values がない");
            return 0;
        }
    }



    int blackScore = 0;
    int whiteScore = 0;

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            if (board[i][j] == 0) {
                /// TODO: 全部見る必要はない
                char tmpColor = 0;
                for (auto dir : directions) {
                    int nx = j + dir.first;
                    int ny = i + dir.second;
                    if (board[ny][nx] == 1) {
                        assert(tmpColor != 2);
                        tmpColor = 1;
                    }
                    else if (board[ny][nx] == 2) {
                        assert(tmpColor != 1);
                        tmpColor = 2;
                    }
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


int MonteCarloTreeSearch()
{
    json j = json::parse("[[0, 0, 2, 2, 2, 1, 0, 0, 0], [0, 0, 0, 2, 1, 1, 1, 0, 0], [0, 0, 2, 2, 2, 2, 1, 1, 0], [0, 0, 0, 2, 1, 2, 1, 1, 0], [0, 2, 2, 2, 1, 2, 2, 1, 2], [0, 1, 2, 1, 1, 2, 1, 2, 0], [0, 2, 1, 1, 1, 1, 1, 0, 1], [0, 2, 2, 2, 2, 2, 1, 1, 2], [0, 0, 0, 0, 0, 2, 1, 0, 0]]");
    vector<vector<char>> v = j;

    goBoard* root(new goBoard(v, 1));


    vector<tuple<char, char, char>> legalMoves = root->GenAllLegalMoves();

    for (auto [y, x, t] : legalMoves) {
        goBoard* tmp = root->PutStone(y, x, t);
    }




    for (auto x = *begin(root->ucts); get<0>(x) <= 0.0; x = *begin(root->ucts)) {
        ++cnt;  //////////

        if (get<0>(x) != 0.0) break;

        double rslt = dfs(root->childrens[get<3>(x)]);

        // print("rslt", rslt);////////

        int win = 0;
        if (rslt > 0) {
            win = 1;
        }
        else if (rslt < 0) {
            win = 0;
        }

        int numWin = get<2>(x) + win;
        int numVisit = get<1>(x) + 1;
        ++root->numVisits;

        // cerr << "numWin: " << numWin << endl;
        // cerr << "numVisit: " << numVisit << endl;
        // cerr << "root->numVisits: " << root->numVisits << endl;

        double uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);

        // print("uct", uct);


        root->ucts.erase(x);

        root->ucts.insert(make_tuple(uct, numVisit, numWin, get<3>(x)));
    }


    for (; cnt < 300; ++cnt) {
        auto x = *rbegin(root->ucts);
        auto [uct, numWin, numVisit, move] = x;

        goBoard* tmpp = root->PutStone(get<0>(move), get<1>(move), root->teban);

        double rslt = dfs(tmpp);

        delete root->childrens[move];



        int win = 0;
        if (rslt > 0) {
            win = 1;
        }
        else if (rslt < 0) {
            win = 0;
        }

        numWin += win;
        numVisit += 1;
        ++root->numVisits;

        uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);
        root->ucts.erase(x);
        auto tmp = make_tuple(uct, numVisit, numWin, move);
        root->ucts.insert(tmp);
        print(tmp, rslt);
    }


    print("end");
    for (auto x : root->ucts) {
        print(x);
    }


    auto ans = *rbegin(root->ucts);

    for (auto x : root->ucts) {
        if (get<1>(x) > get<1>(ans)) {
            ans = x;
        }
        else if (get<1>(x) == get<1>(ans) && get<0>(x) >= get<0>(ans)) {
            ans = x;
        }
    }
    print("ans", ans);

    root->PrintBoard(0b1);

    root->SucceedRoot(rootPtr, get<3>(ans));
    root = rootPtr;


    legalMoves = root->GenAllLegalMoves();

    for (auto [y, x, t] : legalMoves) {
        goBoard* tmp = root->PutStone(y, x, t);
    }
    for (auto x = *begin(root->ucts); get<0>(x) <= 0.0; x = *begin(root->ucts)) {
        ++cnt;  //////////

        if (get<0>(x) != 0.0) break;

        double rslt = dfs(root->childrens[get<3>(x)]);

        // print("rslt", rslt);////////

        int win = 0;
        if (rslt > 0) {
            win = 1;
        }
        else if (rslt < 0) {
            win = 0;
        }

        int numWin = get<2>(x) + win;
        int numVisit = get<1>(x) + 1;
        ++root->numVisits;

        // cerr << "numWin: " << numWin << endl;
        // cerr << "numVisit: " << numVisit << endl;
        // cerr << "root->numVisits: " << root->numVisits << endl;

        double uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);

        // print("uct", uct);


        root->ucts.erase(x);

        root->ucts.insert(make_tuple(uct, numVisit, numWin, get<3>(x)));
    }

    for (; cnt < 300; ++cnt) {
        auto x = *rbegin(root->ucts);
        auto [uct, numWin, numVisit, move] = x;

        goBoard* tmpp = root->PutStone(get<0>(move), get<1>(move), root->teban);

        double rslt = dfs(tmpp);

        delete root->childrens[move];



        int win = 0;
        if (rslt > 0) {
            win = 1;
        }
        else if (rslt < 0) {
            win = 0;
        }

        numWin += win;
        numVisit += 1;
        ++root->numVisits;

        uct = (double)numWin / (double)numVisit + sqrt(2 * log(root->numVisits) / (double)numVisit);
        root->ucts.erase(x);
        auto tmp = make_tuple(uct, numVisit, numWin, move);
        root->ucts.insert(tmp);
        print(tmp, rslt);
    }


    print("end");
    for (auto x : root->ucts) {
        print(x);
    }


    ans = *rbegin(root->ucts);

    for (auto x : root->ucts) {
        if (get<1>(x) > get<1>(ans)) {
            ans = x;
        }
        else if (get<1>(x) == get<1>(ans) && get<0>(x) >= get<0>(ans)) {
            ans = x;
        }
    }
    print("ans", ans);

    root->PrintBoard(0b1);

    cerr << rootPtr << endl;
    root->SucceedRoot(rootPtr, get<3>(ans));
    cerr << rootPtr << endl;


    return 0;
}


int PutStoneCnt = 0;

int suiron(int n)
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();

    print("build end");  ////////////////////

    auto saiki = [tensorRT](auto self, goBoard* ptr) -> tuple<int, float, float, float>
    {
        // print("moveCnt", ptr->moveCnt);  ////////////////
        int color = ptr->teban;

        if (ptr->isEnded) {
            double rslt = ptr->CountResult();
            if (rslt == 0) {
                return make_tuple(color, 0.0, 1.0, 0.0);
            }
            if ((color == 1 && rslt > 0) || (color == 2 && rslt < 0)) {
                return make_tuple(color, 1.0, 0.0, 0.0);
            }
            return make_tuple(color, 0.0, 0.0, 1.0);
        }

        assert(ptr->ucts.size());

        pair<char, char> nextMove = get<3>(*rbegin(ptr->ucts));

        if (!ptr->childrens.count(nextMove)) {
            // print("PutStoneCnt", ++PutStoneCnt);
            goBoard* nextPtr = ptr->PutStone(nextMove.first, nextMove.second, color);

            int nextColor = nextPtr->teban;

            if (nextPtr->isEnded) {
                double rslt = nextPtr->CountResult();
                if (rslt == 0) {
                    return make_tuple(nextColor, 0.0, 1.0, 0.0);
                }
                if ((nextColor == 1 && rslt > 0) || (nextColor == 2 && rslt < 0)) {
                    return make_tuple(nextColor, 1.0, 0.0, 0.0);
                }
                return make_tuple(nextColor, 0.0, 0.0, 1.0);
            }

            return nextPtr->ExpandNode(tensorRT);
        }


        tuple<int, float, float, float> returnData = self(self, ptr->childrens[nextMove]);

        ptr->UpdateUcts(returnData, nextMove);

        return returnData;
    };

    rootPtr = new goBoard();

    rootPtr->ExpandNode(tensorRT);

    // print(rootPtr->policys);
    // print(rootPtr->values);

    float sum = 0.0;
    for (auto x : rootPtr->policys) {
        sum += x.second;
    }
    // print("policyssum:", sum);

    sum = 0.0;
    for (auto x : rootPtr->values) {
        sum += x;
    }
    // print("valuessum:", sum);

    // print("ucts:", rootPtr->ucts);

    int saikiCnt = 0;
    rep (n) {
        // print("saikiCnt:", saikiCnt++);  ////////////////
        saiki(saiki, rootPtr);
    }


    // print("ucts:", rootPtr->ucts);

    // pair<char, char> ans;
    // int ansCnt = -1;
    // for (auto x : rootPtr->ucts) {
    //     if (get<1>(x) >= ansCnt) {
    //         if (get<1>(x) == ansCnt && get<0>(x) < get<0>(ans)) {
    //             continue;
    //         }
    //         ansCnt = get<1>(x);
    //         ans = get<3>(x);
    //     }
    // }
    // print("ans:", ans);



    goBoard* tmp = rootPtr;
    while (true) {
        tmp->PrintBoard(0b1);
        print();
        tmp->PrintBoard(1 << 31);
        print();
        tmp->PrintBoard(1 << 30);
        print();
        tmp->PrintBoard(1 << 29);
        print();
        tmp->PrintBoard(1 << 28);
        print();
        tmp->PrintBoard(1 << 27);
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


    return 0;
}

int Test()
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();
    json j = json::parse("[[0, 0, 2, 2, 2, 1, 0, 0, 0], [0, 0, 0, 2, 1, 1, 1, 0, 0], [0, 0, 2, 2, 2, 2, 1, 1, 0], [0, 0, 0, 2, 1, 2, 1, 1, 0], [0, 2, 2, 2, 1, 2, 2, 1, 2], [0, 1, 2, 1, 1, 2, 1, 2, 0], [0, 2, 1, 1, 1, 1, 1, 0, 1], [0, 2, 2, 2, 2, 2, 1, 1, 2], [0, 0, 0, 0, 0, 2, 1, 0, 0]]");
    vector<vector<char>> v = j;

    rootPtr = new goBoard(v, 1);

    rootPtr->PrintBoard(0b1);
    rootPtr->ExpandNode(tensorRT);
    return 0;
}



// ループを制御するためのフラグ
std::atomic<bool> running(true);

void SearchLoop(TensorRTOnnxIgo& tensorRT)
{
    auto saiki = [tensorRT](auto self, goBoard* ptr) -> tuple<int, float, float, float>
    {
        int color = ptr->teban;

        if (ptr->isEnded) {
            double rslt = ptr->CountResult();
            if (rslt == 0) {
                return make_tuple(color, 0.0, 1.0, 0.0);
            }
            if ((color == 1 && rslt > 0) || (color == 2 && rslt < 0)) {
                return make_tuple(color, 1.0, 0.0, 0.0);
            }
            return make_tuple(color, 0.0, 0.0, 1.0);
        }

        assert(ptr->ucts.size());

        pair<char, char> nextMove = get<3>(*rbegin(ptr->ucts));

        if (!ptr->childrens.count(nextMove)) {
            goBoard* nextPtr = ptr->PutStone(nextMove.first, nextMove.second, color);

            int nextColor = nextPtr->teban;

            if (nextPtr->isEnded) {
                double rslt = nextPtr->CountResult();
                if (rslt == 0) {
                    return make_tuple(nextColor, 0.0, 1.0, 0.0);
                }
                if ((nextColor == 1 && rslt > 0) || (nextColor == 2 && rslt < 0)) {
                    return make_tuple(nextColor, 1.0, 0.0, 0.0);
                }
                return make_tuple(nextColor, 0.0, 0.0, 1.0);
            }

            return nextPtr->ExpandNode(tensorRT);
        }


        tuple<int, float, float, float> returnData = self(self, ptr->childrens[nextMove]);

        ptr->UpdateUcts(returnData, nextMove);

        return returnData;
    };


    while (running.load()) {
        saiki(saiki, rootPtr);
    }
}

int Gpt()
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    rootPtr = new goBoard();

    rootPtr->ExpandNode(tensorRT);

    int saikiCnt = 0;

    // 探索用のスレッドを開始
    std::thread searchThread(SearchLoop, std::ref(tensorRT));

    std::string input;
    // 標準入力を監視
    while (std::getline(std::cin, input)) {
        // vector<string> inputs = split(input, ' ');
        if (input == "list_commands") {
            
        }
        else if (input == "boardsize") {
            
        }
        else if (input == "clear_board") {

        }
        else if (input == "komi") {

        }
        else if (input == "play") {

        }
        else if (input == "genmove") {

        }
        else if (input == "quit") {
            break;
        }

    }

    // ループを停止
    running.store(false);
    // スレッドの終了を待機
    searchThread.join();
}


int main(int argc, char* argv[])
{
    int n = 1000;
    if (argc == 2) n = stoi(argv[1]);
    // MonteCarloTreeSearch();
    suiron(n);
    // Test();

    return 0;
}





//    1 2 3 4 5 6 7 8 9
//  1 ┌ ● ┬ ┬ ┬ ┬ ○ ● ┐
//  2 ● ● + + + + + ○ ○
//  3 ├ ● + ● ● ● + + ○
//  4 ○ ● + ● + ● + ○ ┤
//  5 ● ● + ● ● + + + ○
//  6 ├ + + + + + + + ┤
//  7 ├ + + + + ○ + + ┤
//  8 ● + + + ● + ○ + ┤
//  9 └ ● ┴ ┴ ┴ ○ ○ ┴ ┘


// Move : 3
// Prisoner(Black) : 0
// Prisoner(White) : 0
//     A B C D E F G H J
//   +-------------------+
//  9| + + + + + + + + + |
//  8| + + + + + + + + + |
//  7| + + + + + + + + + |
//  6| + + + + + + + + + |
//  5| + + + O + @ + + + |
//  4| + + + + + + + + + |
//  3| + + + + + + + + + |
//  2| + + + + + + + + + |
//  1| + + + + + + + + + |
//   +-------------------+





// "legalMoves.size():", 78
// pass: O
// _ O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O _ _ O O O O
// O O O O _ O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// [01/22/2025-01:11:22] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +0, now: CPU 0, GPU 1 (MiB)
// "tmpPolicy.size():", 82
// -3.0643 -2.5386 -0.6482 -1.9070 -1.9800 -1.7157 -2.3123 -2.8675 -2.1690
// -1.6564 +2.0302 +0.2747 -0.3175 -0.6484 -1.2339 -1.6391 -1.0966 -2.2894
// -0.9973 +0.5675 +2.7738 +4.3787 +2.9279 +3.2399 +4.2932 -1.0965 -1.6416
// -1.9561 -0.5473 +2.6993 +7.6882 +8.4930 +1.6248 +3.7076 -0.1639 -1.8169
// -1.4363 -0.9100 +1.4575 +0.2655 +2.3247 +0.8941 +3.2148 +1.4013 -1.8689
// -1.7469 -1.0087 +3.7042 +7.8111 +0.4135 +0.2033 +3.8415 -1.0274 -1.8955
// -1.8853 -1.6004 +4.9563 +0.5469 +0.9036 +3.8731 +4.1513 -1.6426 -2.0869
// -2.4943 -1.5963 -1.7056 -1.0420 +1.8147 -0.8280 -0.9790 -1.6516 -2.0485
// -2.1292 -2.7262 -2.3229 -2.4141 -2.1758 -1.7866 -1.9552 -2.8590 -1.3960
// -0.1663
// "values.size():", 3
// "values:", [3.2885046005249023,-5.2117695808410645,1.8991395235061646]
// "policys.size():", 78
// ####   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +1   +0   +0   +0   +0   +0   +0   +0
//   +0   +0   +2  +13   +3   +4  +12   +0   +0
//   +0   +0   +2 +357 +800   +0   +6   +0   +0
//   +0   +0   +0 #### @@@@   +0   +4   +0   +0
//   +0   +0   +6 +404 @@@@   +0   +7   +0   +0
//   +0   +0  +23   +0   +0   +7  +10   +0   +0
//   +0   +0   +0   +0   +1   +0   +0   +0   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
// "pass:", 0.00013886754459235817
// "values:", [0.8003605008125305,0.00016280340787488967,0.19947664439678192]
// "inputColor:", 1
// "inputWin:", 0.8003605008125305
// "inputDraw:", 0.00016280340787488967
// "inputLose:", 0.19947664439678192
// "teban:", 1
// "inputColor:", 1
// "inputWin:", 0.8003605008125305
// "inputDraw:", 0.00016280340787488967
// "inputLose:", 0.19947664439678192
// "teban:", 2
// "inputColor:", 1
// "inputWin:", 0.8003605008125305
// "inputDraw:", 0.00016280340787488967
// "inputLose:", 0.19947664439678192
// "teban:", 1
// "ucts:", [[0.6092992646916244,372,134.11196899414063,[3,9]],[0.6093166666034534,423,159.05331420898438,[1,4]],[0.6093407986958637,333,115.34868621826172,[9,6]],[0.6094098131440249,484,189.39048767089844,[8,3]],[0.6094275614198674,322,110.13223266601563,[4,1]],[0.6094289947593488,338,117.77027893066406,[3,1]],[0.6094729396226544,407,151.2522735595703,[8,1]],[0.6094737355597452,312,105.39991760253906,[9,4]],[0.609493521319866,301,100.20916748046875,[7,1]],[0.6095109982676179,377,136.618408203125,[1,3]],[0.6095430656354879,349,123.0898208618164,[6,1]],[0.6095765063190435,384,140.04916381835938,[2,9]],[0.6095810383268967,455,175.0069122314453,[1,2]],[0.6096364926603808,287,93.67635345458984,[9,7]],[0.6096478916951087,347,122.1643295288086,[0,0]],[0.6096686226672854,447,171.07321166992188,[1,5]],[0.6096755980209374,575,235.50291442871094,[7,2]],[0.6096955445873538,383,139.60769653320313,[1,6]],[0.609698120237273,369,132.80517578125,[2,1]],[0.6096995967409444,535,215.2029571533203,[2,7]],[0.6097373252628113,424,159.7244110107422,[4,9]],[0.6097451041631885,464,179.56195068359375,[2,8]],[0.6097492988791426,901,405.35345458984375,[6,2]],[0.6097570167000657,403,149.40573120117188,[8,8]],[0.6097677749938692,447,171.11753845214844,[8,9]],[0.6097689734376743,489,192.06930541992188,[2,2]],[0.6097771025996042,359,127.99408721923828,[7,9]],[0.6097951834633637,812,358.42156982421875,[6,8]],[0.6097961579189812,564,229.97036743164063,[8,4]],[0.6097995001412649,1146,536.3932495117188,[3,7]],[0.6098347675112261,287,93.7332534790039,[1,9]],[0.6098420271976313,591,243.76622009277344,[3,8]],[0.609846414594648,486,190.60362243652344,[3,2]],[0.6098495429715789,364,130.43841552734375,[5,1]],[0.6098508341498048,377,136.74652099609375,[9,8]],[0.609852830109735,336,116.95521545410156,[9,3]],[0.6098545473161895,3298,1735.7392578125,[3,5]],[0.6098586778497359,479,187.10513305664063,[5,9]],[0.609865531170848,368,132.38206481933594,[9,2]],[0.6098691087468326,870,389.054931640625,[4,8]],[0.6098713133164543,378,137.24063110351563,[1,7]],[0.6098735461045286,420,157.80982971191406,[8,2]],[0.6098768288611798,341,119.36064910888672,[9,5]],[0.6098779013295872,1127,526.2476196289063,[2,5]],[0.6098832343181111,2426,1243.236328125,[4,3]],[0.609886790275616,324,111.23289489746094,[9,9]],[0.6098923286356993,4228,2266.620849609375,[5,4]],[0.6098926617777454,2796,1451.5361328125,[5,7]],[0.6098936892372784,414,154.86386108398438,[1,8]],[0.6099063210888318,2397,1227.0218505859375,[4,7]],[0.6099127252373407,844,375.36602783203125,[2,4]],[0.6099158187841274,3640,1930.59716796875,[6,4]],[0.6099162592891505,4297,2306.270751953125,[4,5]],[0.6099194190492148,2187,1109.496826171875,[6,7]],[0.6099203477008661,591,243.8125,[7,8]],[0.6099205406865276,387,141.6444549560547,[6,9]],[0.6099257661952737,898,403.92266845703125,[2,6]],[0.6099288695098688,626,261.7606201171875,[2,3]],[0.6099293294766515,4215,2259.328125,[5,6]],[0.6099319482281041,1397,672.7293090820313,[5,8]],[0.609934388628046,3342,1761.0074462890625,[6,6]],[0.6099355806320299,754,328.133056640625,[4,2]],[0.6099400594716415,2996,1564.7388916015625,[5,3]],[0.609941793243402,1135,530.62841796875,[3,3]],[0.6099462165735812,4693,2533.763916015625,[5,5]],[0.6099466172258136,2740,1420.0836181640625,[3,4]],[0.609947604185979,282,91.42710876464844,[1,1]],[0.6099490094837969,2035,1024.7877197265625,[7,4]],[0.6099500259319357,2612,1347.956298828125,[3,6]],[0.6099514513918534,455,175.1754608154297,[8,7]],[0.6099516056217734,2125,1074.953857421875,[7,6]],[0.6099562107257968,990,452.8798828125,[5,2]],[0.6099563569077573,2824,1467.5257568359375,[7,5]],[0.6099567673480758,956,434.757080078125,[7,7]],[0.6099579623292359,3980,2124.91748046875,[6,5]],[0.6099606234066246,2439,1250.72119140625,[6,3]],[0.6099706514236913,3594,1904.5731201171875,[4,6]],[0.6099730601074095,3808,2026.6756591796875,[4,4]],[0.609973765479847,1063,491.9580078125,[8,5]],[0.6099749389925556,961,437.4368591308594,[7,3]],[0.6099765023750792,627,262.3045654296875,[8,6]],[0.6099790554441333,391,143.6204376220703,[9,1]]]
// "ans:", [5,5]
// -1 -1 +1
// moveCnt: +0, teban: +1
//    1 2 3 4 5 6 7 8 9
// +1 ┌ ┬ ┬ ┬ ┬ ┬ ┬ ┬ ┐
// +2 ├ + + + + + + + ┤
// +3 ├ + + + + + + + ┤
// +4 ├ + + + + + + + ┤
// +5 ├ + + + + + + + ┤
// +6 ├ + + + + + + + ┤
// +7 ├ + + + + + + + ┤
// +8 ├ + + + + + + + ┤
// +9 └ ┴ ┴ ┴ ┴ ┴ ┴ ┴ ┘

// "policys.size():", 82
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +1   +0   +0   +0   +0   +0   +0   +0
//   +0   +0  +12  +26   +9  +25  +12   +0   +0
//   +0   +0  +26  +97 +103  +95  +26   +0   +0
//   +0   +0   +9 +103 +512 +103   +9   +0   +0
//   +0   +0  +25  +94 +101  +95  +26   +0   +0
//   +0   +0  +11  +25   +9  +24  +11   +0   +0
//   +0   +1   +0   +0   +0   +0   +0   +1   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
// "pass:", 0.00010275642125634477
// "values:", [0.5129799842834473,0.0004004337824881077,0.4866195619106293]

// "childrens.size():", 82
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O

// "ucts.size():", 82
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61 +0.61
// pass: +0.61

// y:  5
// x: 5
// +5 +5 +2
// moveCnt: +1, teban: +2
//    1 2 3 4 5 6 7 8 9
// +1 ┌ ┬ ┬ ┬ ┬ ┬ ┬ ┬ ┐
// +2 ├ + + + + + + + ┤
// +3 ├ + + + + + + + ┤
// +4 ├ + + + + + + + ┤
// +5 ├ + + + ● + + + ┤
// +6 ├ + + + + + + + ┤
// +7 ├ + + + + + + + ┤
// +8 ├ + + + + + + + ┤
// +9 └ ┴ ┴ ┴ ┴ ┴ ┴ ┴ ┘

// "policys.size():", 81
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +0   +0   +0   +1   +0   +0   +0   +0
//   +0   +0  +46  +80 +524  +78  +47   +0   +0
//   +0   +0  +82  +12  +23  +12  +82   +0   +0
//   +0   +1 +540  +23 ####  +24 +520   +1   +0
//   +0   +0  +77  +12  +24  +15  +80   +0   +0
//   +0   +0  +50  +82 +519  +82  +51   +0   +0
//   +0   +0   +0   +0   +1   +0   +0   +0   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
// "pass:", 0.00042700927588157356
// "values:", [0.5404638648033142,0.0004166991566307843,0.45911943912506104]

// "childrens.size():", 81
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O X O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O
// O O O O O O O O O

// "ucts.size():", 81
// +0.97 +0.98 +0.97 +0.98 +0.98 +0.98 +0.98 +0.98 +0.97
// +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.97 +0.98 +0.98
// +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98
// +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98
// +0.98 +0.98 +0.98 +0.98 +0.00 +0.98 +0.98 +0.98 +0.98
// +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98
// +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98
// +0.98 +0.98 +0.97 +0.98 +0.98 +0.98 +0.98 +0.98 +0.98
// +0.97 +0.98 +0.98 +0.98 +0.98 +0.97 +0.98 +0.98 +0.98
// pass: +0.97

// y: 3
// x: 5
// +3 +5 +1
// moveCnt: +2, teban: +1
//    1 2 3 4 5 6 7 8 9
// +1 ┌ ┬ ┬ ┬ ┬ ┬ ┬ ┬ ┐
// +2 ├ + + + + + + + ┤
// +3 ├ + + + ○ + + + ┤
// +4 ├ + + + + + + + ┤
// +5 ├ + + + ● + + + ┤
// +6 ├ + + + + + + + ┤
// +7 ├ + + + + + + + ┤
// +8 ├ + + + + + + + ┤
// +9 └ ┴ ┴ ┴ ┴ ┴ ┴ ┴ ┘

// "policys.size():", 80
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +0   +0   +0  +30   +0   +0   +0   +0
//   +0   +0   +4 +524 @@@@ +538   +5   +0   +0
//   +0   +0 +177  +70  +12  +66 +181   +0   +0
//   +0   +0  +85   +0 ####   +0  +86   +0   +0
//   +0   +0   +7   +1   +0   +2   +6   +0   +0
//   +0   +0   +0  +52 +428  +49   +0   +0   +0
//   +0   +0   +0   +0   +9   +0   +0   +0   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
// "pass:", 0.0002202322648372501
// "values:", [0.5387895703315735,0.0005389123107306659,0.4606715142726898]

// "childrens.size():", 65
// X X O O O O O X X
// O O O O O O O O X
// O O O O X O O O X
// O O O O O O O O X
// O O O O X O O O O
// O O O O O O O O O
// X O O O O O O O O
// X X O O O O O X O
// X X X O O O O X O

// "ucts.size():", 80
// +2.89 +2.89 +2.22 +2.24 +2.25 +2.17 +2.16 +2.89 +2.89
// +2.16 +2.22 +2.24 +2.21 +2.27 +2.23 +2.19 +2.17 +2.89
// +2.17 +2.22 +2.23 +2.56 +0.00 +2.55 +2.23 +2.16 +2.89
// +2.17 +2.23 +2.35 +2.38 +2.34 +2.35 +2.34 +2.18 +2.89
// +2.17 +2.18 +2.32 +2.24 +0.00 +2.25 +2.33 +2.19 +2.15
// +2.17 +2.18 +2.24 +2.27 +2.23 +2.24 +2.20 +2.19 +2.16
// +2.89 +2.16 +2.17 +2.24 +2.51 +2.24 +2.18 +2.15 +2.89
// +2.89 +2.89 +2.22 +2.21 +2.23 +2.23 +2.14 +2.89 +2.16
// +2.89 +2.89 +2.89 +2.14 +2.14 +2.16 +2.14 +2.89 +2.13
// pass: +2.14

// y: 3
// x: 4
// +3 +4 +2
// moveCnt: +3, teban: +2
//    1 2 3 4 5 6 7 8 9
// +1 ┌ ┬ ┬ ┬ ┬ ┬ ┬ ┬ ┐
// +2 ├ + + + + + + + ┤
// +3 ├ + + ● ○ + + + ┤
// +4 ├ + + + + + + + ┤
// +5 ├ + + + ● + + + ┤
// +6 ├ + + + + + + + ┤
// +7 ├ + + + + + + + ┤
// +8 ├ + + + + + + + ┤
// +9 └ ┴ ┴ ┴ ┴ ┴ ┴ ┴ ┘

// "policys.size():", 79
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +0   +0 +102   +0   +0   +0   +0   +0
//   +0   +0   +1 #### @@@@   +0   +0   +0   +0
//   +0   +0   +0 +508   +3   +0   +2   +0   +0
//   +0   +0   +0   +0 ####   +0   +0   +0   +0
//   +0   +0   +1   +0   +1   +0   +0   +0   +0
//   +0   +0   +4  +21  +69   +5   +1   +0   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
//   +0   +0   +0   +0   +0   +0   +0   +0   +0
// "pass:", 1.6081046851468273e-05
// "values:", [0.5089550018310547,0.0005318978219293058,0.49051305651664734]

// "childrens.size():", 1
// X X X X X X X X X
// X X X X X X X X X
// X X X X X X X X X
// X X X O X X X X X
// X X X X X X X X X
// X X X X X X X X X
// X X X X X X X X X
// X X X X X X X X X
// X X X X X X X X X

// "ucts.size():", 79
// +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +3.06 +2.96 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +0.00 +0.00 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +3.47 +2.96 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +2.96 +0.00 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +2.98 +3.03 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96
// +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96 +2.96
// pass: +2.96