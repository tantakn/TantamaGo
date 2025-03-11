
#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
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

pair<vector<float>, vector<float>> Infer(vector<vector<vector<float>>> inputPlane, TensorRTOnnxIgo& tensorRT)
{
    vector<float> outputPolicy(BOARDSIZE * BOARDSIZE + 1, 0.0);
    vector<float> outputValue;

    tensorRT.infer(inputPlane, outputPolicy, outputValue);

    return make_pair(outputPolicy, outputValue);
}


void SearchLoop(goBoard* rootPtr, TensorRTOnnxIgo& tensorRT, atomic<bool>& running)
{
    // (PutStone or new) -> (ExpandNode) -> (PutStone) ...
    // rootPtr new -> rootPtr.ExpandNode をしておく。
    // ptr0 -> ... -> if !ptr3->children.count(nextMove) -> PutStone -> ExpandNode -> ptr2 の ucts を更新 -> ... -> ptr0 の ucts を更新。 という流れを繰り返す。
    // ExpandNode で 合法手に対して ucts.insert される。
    // PutStone の手が連続2回目のpassのとき isEnded = true になる。
    auto saiki = [&tensorRT](auto self, goBoard* ptr) -> tuple<int, float, float, float>
    {
        int color = ptr->teban;

        if (ptr->isEnded) {
            double rslt = ptr->CountResult();
            if (rslt == 0) {
                return make_tuple(color, 0.0, 1.0, 0.0);
            }
            if ((color == 1 && rslt > 0) || (color == 2 && rslt < 0)) {
                return make_tuple(color, 0.0, 0.0, 1.0);
            }
            return make_tuple(color, 1.0, 0.0, 0.0);
        }

        lock_guard<recursive_mutex> lock(ptr->uctsMutex);

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
                    return make_tuple(color, 0.0, 0.0, 1.0);
                }
                return make_tuple(color, 1.0, 0.0, 0.0);
            }

            return nextPtr->ExpandNode(Infer(ptr->MakeInputPlane(), tensorRT));
        }


        tuple<int, float, float, float> returnData = self(self, ptr->childrens[nextMove]);

        ptr->UpdateUcts(returnData, nextMove);

        return returnData;
    };


    while (running.load()) {
        if (rootPtr->isEnded) {
            break;
        }
        saiki(saiki, rootPtr);
    }

    return;
}

string Gpt(const string input, goBoard*& rootPtr, TensorRTOnnxIgo& tensorRT, thread& searchThread, atomic<bool>& running, int thinkTime = 1, bool ponder = false)
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
        rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));

        running.store(true);
        searchThread = thread(SearchLoop, rootPtr, ref(tensorRT), ref(running));

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
            searchThread.join();

            if (rootPtr->childrens.size() == 0) {
                rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));
            }
            rootPtr = rootPtr->SucceedRoot(rootPtr, {y, x});
            if (rootPtr->isEnded) {
                goto GOTO_GPT_SEND;
            }

            running.store(true);
            searchThread = thread(SearchLoop, rootPtr, ref(tensorRT), ref(running));

            output = "=";
        }
    }
    else if (commands[0] == "genmove") {
        if (rootPtr->isEnded) {
            output = "game has already ended";
            goto GOTO_GPT_SEND;
        }

        sleep(thinkTime);  ///////////////
        running.store(false);
        searchThread.join();

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
            rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));
        }
        rootPtr = rootPtr->SucceedRoot(rootPtr, move);
        if (rootPtr->isEnded) {  ////////////????
            goto GOTO_GPT_SEND;
        }

        running.store(true);
        searchThread = thread(SearchLoop, rootPtr, ref(tensorRT), ref(running));
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
        searchThread.join();

        output = "=";
        rootPtr->PrintBoard(0b1);

        running.store(true);
        searchThread = thread(SearchLoop, rootPtr, ref(tensorRT), ref(running));
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

    rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));


    int saikiCnt = 0;
    // ループを制御するためのフラグ
    std::atomic<bool> running(true);
    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT), ref(running));

    sleep(n);
    running.store(false);
    searchThread.join();


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
        rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));
    }
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





int PlayWithGpt()
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));

    int saikiCnt = 0;

    // ループを制御するためのフラグ
    std::atomic<bool> running(true);
    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT), ref(running));

    string input;
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        output = Gpt(input, rootPtr, tensorRT, searchThread, ref(running));
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
    int port = 8000;
    cout << "port << ";
    cin >> port;


    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, tensorRTModelPath));

    tensorRT.build();

    print("build end");  ////////////////////


    goBoard* rootPtr = nullptr;
    rootPtr = new goBoard();

    rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));

    int saikiCnt = 0;

    // ループを制御するためのフラグ
    std::atomic<bool> running(true);
    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT), ref(running));

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

        if (input.substr(0, 4) == "genm" || input.substr(0, 4) == "play") {
            cout << "\n--------------------\n"
                 << "rootPtr: " << rootPtr << endl;  //////////////////////////
            print();
            rootPtr->PrintBoard(1 << 26);  //////////////////
            print();
            rootPtr->PrintBoard(1 << 28);  //////////////////
            print();
            rootPtr->PrintBoard(1 << 31);
            print();
            rootPtr->PrintBoard(1 << 27);
            print();
            rootPtr->PrintBoard(1 << 29);
            print();
            rootPtr->PrintBoard(0b1);
            print();
        }


        cerr << "recv data: " << buf << endl;  /////////////////////

        output = Gpt(buf, rootPtr, tensorRT, searchThread, ref(running), thinkTime, false);

        if (output == "quit") {
            output = "=";
            write(client_sockfd, output.c_str(), output.length());
            break;
        }

        cerr << "send data: " << output << endl;  /////////////////////

        output += "\n";
        write(client_sockfd, output.c_str(), output.length());

        // Clear the buffer after sending the data
        memset(buf, 0, sizeof(buf));

        sleep(1);
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
    rootPtr->ExpandNode(Infer(rootPtr->MakeInputPlane(), tensorRT));
    int saikiCnt = 0;

    // ループを制御するためのフラグ
    std::atomic<bool> running(true);
    // 探索用のスレッドを開始
    thread searchThread(SearchLoop, rootPtr, ref(tensorRT), ref(running));


    string input = "";
    string output = "";
    // 標準入力を監視
    while (getline(cin, input)) {
        output = Gpt(input, rootPtr, tensorRT, searchThread, ref(running));
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