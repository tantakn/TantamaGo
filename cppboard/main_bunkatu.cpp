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


// (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ g++ -w -Wno-deprecated-declarations -std=c++17 -I"./TensorRT/common" -I"./TensorRT/utils" -I"./TensorRT" -I"/usr/local/cuda/include" -I"./TensorRT/include" -D_REENTRANT -DTRT_STATIC=0 -g main_bunkatu.cpp globals.cpp ./goBoard_bunkatu.cpp ./TensorRT/common/bfloat16.cpp ./TensorRT/common/getOptions.cpp ./TensorRT/common/logger.cpp ./TensorRT/common/sampleDevice.cpp ./TensorRT/common/sampleEngines.cpp ./TensorRT/common/sampleInference.cpp ./TensorRT/common/sampleOptions.cpp ./TensorRT/common/sampleReporting.cpp ./TensorRT/common/sampleUtils.cpp ./TensorRT/utils/fileLock.cpp ./TensorRT/utils/timingCache.cpp -o goBoard_bunkatu -L"/usr/local/cuda/lib64" -Wl,-rpath-link="/usr/local/cuda/lib64" -L"./TensorRT/lib" -Wl,-rpath-link="./TensorRT/lib" -L"./TensorRT/bin" -Wl,--start-group -lnvinfer -lnvinfer_plugin -lnvonnxparser -lcudart -lrt -ldl -lpthread -Wl,--end-group -Wl,--no-relax

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



#ifndef tensorRTigo_cpp_INCLUDED
#include "./tensorRTigo.cpp"
#define tensorRTigo_cpp_INCLUDED
#endif

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

// double dfs(goBoard* ptr)
// {
//     // print("dfs", cnt);////////////////
//     if (ptr->isEnded) {
//         return ptr->CountResult();
//     }

//     tuple<char, char, char> legalMove = ptr->GenRandomMove();

//     double tmp = dfs(ptr->PutStone(get<0>(legalMove), get<1>(legalMove), get<2>(legalMove)));
//     // if (ptr->parent->isRoot) {
//     //     for (auto x : ptr->childrens) {
//     //         delete(x.second);
//     //     }
//     // }
//     // if (!ptr->isRoot){
//     //     delete ptr;
//     // }
//     return tmp;
// }


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




pair<vector<float>, vector<float>> Infer(TensorRTOnnxIgo& tensorRT, goBoard* ptr)
{
    vector<float> tmpPolicys(BOARDSIZE * BOARDSIZE + 1, 0);

    vector<float> tmpValues(3, 0);

    tensorRT.infer(ptr->MakeInputPlane(), tmpPolicys, tmpValues);

    return {tmpPolicys, tmpValues};
}



// ループを制御するためのフラグ
std::atomic<bool> running(true);

void SearchLoop(goBoard* rootPtr, TensorRTOnnxIgo& tensorRT)
{
    // leaf（ExpandNode で作ったノード・isEndedのノード）ではuctの更新はしない。
    while (running.load()) {
        if (rootPtr->isEnded) {
            break;
        }
        if (rootPtr->numVisits > visitMax) {
            sleep(0.1);
        }

        // 潜る探索ループ
        goBoard* ptr = rootPtr;
        tuple<int, float, float, float> leafRslt;
        while (true) {
            char color = ptr->teban;

            // もし終局（直前2手がパス）なら結果を leafRslt に入れてbreak
            if (ptr->isEnded) {
                if (ptr->ucts.size() == 0) {
                    ptr->ExpandNode(Infer(tensorRT, ptr));
                    // ptr->ExpandNode(tensorRT);
                }
                double tmpRslt = ptr->CountResult(false);
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

                // 葉ノードがisendeedのとき、valueでなく勝敗で評価を行う。
                if (nextPtr->isEnded) {
                    if (nextPtr->ucts.size() == 0) {
                        nextPtr->ExpandNode(Infer(tensorRT, ptr));
                        // ptr->ExpandNode(tensorRT);
                    }
#ifdef dbg_flag
                    ++endCnt;
#endif
                    double tmpRslt = nextPtr->CountResult(false);
                    if (tmpRslt == 0) {
                        leafRslt = make_tuple(nextColor, 0.0, 1.0, 0.0);
                    }
                    else if ((nextColor == 1 && tmpRslt > 0) || (nextColor == 2 && tmpRslt < 0)) {
                        leafRslt = make_tuple(nextColor, 0.0, 0.0, 1.0);
                    }
                    else {
                        leafRslt = make_tuple(nextColor, 1.0, 0.0, 0.0);
                    }
                    break;
                }

                nextPtr->ExpandNode(Infer(tensorRT, nextPtr));
                // nextPtr->ExpandNode(tensorRT);
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
    stringstream ss{input};
    string s;
    vector<string> commands;

    string output;
    // スペース（' '）で区切って，vに格納
    while (getline(ss, s, ' ')) {
        commands.push_back(s);
    }


    if (commands[0] == "list_commands") {
        output = "=list_commands\nname\nboardsize\nclear_board\nkomi\nplay\ngenmove\nquit\nshowboard";
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
        rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
        // rootPtr->ExpandNode(tensorRT);

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
                    cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
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
                    cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    goto GOTO_GPT_SEND;
                }


                if (y < 1 || y > BOARDSIZE) {
                    output = "dismatch_boardsize";
                    cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    print("x, y: ", x, y);                                   //////////////
                    goto GOTO_GPT_SEND;
                }

                if (commands[1] == "black" || commands[1] == "b" || commands[1] == "B") {
                    if (rootPtr->teban != 1) {
                        cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                        output = "dismatch_color";
                        goto GOTO_GPT_SEND;
                    }
                }
                else if (commands[1] == "white" || commands[1] == "w" || commands[1] == "W") {
                    if (rootPtr->teban != 2) {
                        cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                        output = "dismatch_color";
                        goto GOTO_GPT_SEND;
                    }
                }
                else {
                    output = "unknown_command";
                    cerr << "commands[2]==[" << commands[2] << "]" << endl;  ////////////////////
                    goto GOTO_GPT_SEND;
                }
            }

            running.store(false);
            if (searchThread.joinable()) {
                searchThread.join();
            }

            if (debugFlag & 1 << 5) {
                cerr << "\n--------------------\n"
                     << "rootPtr: " << rootPtr << ", teban: " << int(rootPtr->teban) << ", moveCnt: " << rootPtr->moveCnt << endl;  //////////////////////////
#ifdef dbg_flag
                cerr << "expandCnt: " << expandCnt << ", endCnt: " << endCnt << endl;
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

            if (rootPtr->ucts.size() == 0) {
                rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
                // rootPtr->ExpandNode(tensorRT);
            }
            rootPtr = rootPtr->SucceedRoot(rootPtr, {y, x});
            if (rootPtr->ucts.size() == 0 && !rootPtr->isEnded) {
                rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
                // rootPtr->ExpandNode(tensorRT);
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

        sleep(thinkTime);  ////////////////

        if (debugFlag & 1 << 5) {
            cerr << "\n--------------------\n"
                 << "rootPtr: " << rootPtr << ", teban: " << int(rootPtr->teban) << ", moveCnt: " << rootPtr->moveCnt << endl;  //////////////////////////
#ifdef dbg_flag
            cerr << "expandCnt: " << expandCnt << ", endCnt: " << endCnt << endl;
#endif
            print();
            // rootPtr->PrintBoard(0b100);
            // print();
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

        if (rootPtr->ucts.size() == 0) {
            rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
            // rootPtr->ExpandNode(tensorRT);
        }
        rootPtr = rootPtr->SucceedRoot(rootPtr, move);
        if (rootPtr->ucts.size() == 0 && !rootPtr->isEnded) {
            rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
            // rootPtr->ExpandNode(tensorRT);
        }
        if (rootPtr->isEnded) {  ////////////????
            rootPtr->CountResult(true);//////////////////
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

    rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
    // rootPtr->ExpandNode(tensorRT);


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

    if (rootPtr->ucts.size() == 0) {
        rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
        // rootPtr->ExpandNode(tensorRT);
    }
    rootPtr = rootPtr->SucceedRoot(rootPtr, {y, x});
    if (rootPtr->ucts.size() == 0 && !rootPtr->isEnded) {
        rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
        // rootPtr->ExpandNode(tensorRT);
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
    // cerr << "thinkTime << ";
    // cin >> thinkTime;
    int visitMax = 100000;
    // cerr << "visitMax << ";
    // cin >> visitMax;


    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = true;
    args.runInBf16 = true;
    // args.runInInt8 = false;
    // args.runInFp16 = false;
    // args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
    // rootPtr->ExpandNode(tensorRT);

    int saikiCnt = 0;

    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));

    string input;
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        output = Gpt(input, rootPtr, tensorRT, searchThread, thinkTime, true);
        if (output == "exit") {
            output += "\n";
            cout << "=" << endl;
            break;
        }
        output += "\n";
        cout << output << endl;
    }

    return 0;
}



int GptSoket()
{
    int thinkTime = 10;
    cerr << "thinkTime << ";
    cin >> thinkTime;
    // cerr << "visitMax << ";
    // cin >> visitMax;
    int port = 8000;
    cerr << "port << ";
    cin >> port;


    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = true;
    args.runInBf16 = true;
    // args.runInInt8 = false;
    // args.runInFp16 = false;
    // args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();

    print("build end");  ////////////////////


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
    // rootPtr->ExpandNode(tensorRT);

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
        //     cerr << "\n--------------------\n"
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
    args.runInFp16 = true;
    args.runInBf16 = true;
    // args.runInInt8 = false;
    // args.runInFp16 = false;
    // args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    json j = json::parse(s);
    vector<vector<char>> v = j;

    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard(v, teban);

    rootPtr->PrintBoard(0b1);
    rootPtr->ExpandNode(Infer(tensorRT, rootPtr));
    // rootPtr->ExpandNode(tensorRT);
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
