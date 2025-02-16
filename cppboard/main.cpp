#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

#ifndef goBoard_hpp_INCLUDED
#include "goBoard.hpp"
#define goBoard_hpp_INCLUDED
#endif


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

    // float sum = 0.0;
    // for (auto x : rootPtr->policys) {
    //     sum += x.second;
    // }
    // print("policyssum:", sum);

    // sum = 0.0;
    // for (auto x : rootPtr->values) {//??????
    //     sum += x;
    // }
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

    rootPtr->SucceedRoot(rootPtr, {y, x});

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

void SearchLoop(goBoard* rootPtr, TensorRTOnnxIgo& tensorRT)
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
                /// TODO: 正しいか確認
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
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT));

    string input;
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        stringstream ss{input};
        string s;
        vector<string> commands;

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
        else if (commands[0] == "boardsize") {
            output = "=";
        }
        else if (commands[0] == "clear_board") {
            delete rootPtr;
            rootPtr = new goBoard();
            rootPtr->ExpandNode(tensorRT);
            output = "=";
        }
        else if (commands[0] == "komi") {
            output = "=";
        }
        else if (commands[0] == "play") {
            char y, x;

            print(commands[1]);  /////////////////////
            print(commands[2]);  /////////////////////

            if (commands[2][0] >= 'a' && commands[2][0] < 'a' + BOARDSIZE) {
                x = commands[2][0] - 'a' + 1;
            }
            else if (commands[2][0] >= 'A' && commands[2][0] << 'A' + BOARDSIZE) {
                x = commands[2][0] - 'A' + 1;
            }
            else {
                output = "dismatch_boardsize";
                continue;
            }

            if (commands[2].size() == 2) {
                y = commands[2][1] - '0';
            }
            else if (commands[2].size() == 3) {
                y = (commands[2][1] - '0') * 10 + commands[2][2] - '0';
            }
            else {
                output = "unknown_command";
                continue;
            }
            if (y < 1 || y > BOARDSIZE) {
                output = "dismatch_boardsize";
                continue;
            }

            if (commands[1] == "black" || commands[1] == "b") {
                if (rootPtr->teban != 1) {
                    output = "dismatch_color";
                    continue;
                }
            }
            else if (commands[1] == "white" || commands[1] == "w") {
                if (rootPtr->teban != 2) {
                    output = "dismatch_color";
                    continue;
                }
            }
            else {
                output = "unknown_command";
                continue;
            }

            // print("y:", y);  /////////////////////
            // print("x:", x);  /////////////////////

            running.store(false);

            searchThread.join();

            rootPtr->SucceedRoot(rootPtr, {y, x});
            if (rootPtr->isEnded) {
                running.store(false);
                searchThread.join();
                continue;
            }

            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));

            output = "=";
        }
        else if (commands[0] == "genmove") {
            sleep(1);
            running.store(false);
            searchThread.join();

            pair<char, char> move = rootPtr->GetAns();
            // print("move:", move);///////////////

            if (move.first == 0 && move.second == 0) {
                output = "=pass";
            }
            else {
                output = "=";
                output += char(move.second - 1 + 'A');
                output += to_string(move.first);
            }

            rootPtr->SucceedRoot(rootPtr, move);
            if (rootPtr->isEnded) {
                running.store(false);
                searchThread.join();
                continue;
            }

            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT));
        }
        else if (commands[0] == "quit") {
            // ループを停止
            running.store(false);
            // スレッドの終了を待機
            searchThread.join();
            break;
        }
        else if (commands[0] == "showboard") {
            output = "";
            rootPtr->PrintBoard(0b1);
        }
        else {
            output = "unknown_command";
        }

        cout << output << endl;
    }
}


int main(int argc, char* argv[])
{
    int n = 1000;
    if (argc == 2) n = stoi(argv[1]);
    // MonteCarloTreeSearch();
    // suiron(n);
    // Test();

    Gpt();

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
