#define goBoard_cpp_INCLUDED


#ifndef myMacro_hpp_INCLUDED
#include "myMacro.hpp"
#define myMacro_hpp_INCLUDED
#endif

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
    : board(rawBoard), idBoard(rawIdBoard), teban(1), parent(nullptr), isRoot(true), moveCnt(1)
{
    assert(rootPtr == nullptr);
    rootPtr = this;

    libs[-1] = INF;
}

goBoard::goBoard(vector<vector<char>> inputBoard)
    : board(InputBoardFromVec(inputBoard)), idBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0)), teban(1), isRoot(true), moveCnt(1)
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

goBoard* goBoard::SucceedRoot(pair<char, char> move)
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

    vector<tuple<char, char, char>> moves = GenAllLegalMoves();
    // for (auto [y, x, t] : moves) {
    //     childrens[make_pair(y, x)] = new goBoard(*this, y, x, t);
    // }


    vector<float> tmpPolicy(BOARDSIZE * BOARDSIZE + 1, 0.0);

    tensorRT.infer(MakeInputPlane(), tmpPolicy, values);

    // print("tmpPolicy.size():", tmpPolicy.size());//////////////
    // print("tmpPolicy:", tmpPolicy);

    // rep (i, tmpPolicy.size()) {
    //     cout << setprecision(4) << tmpPolicy[i] << " ";
    // }

    // print("values.size():", values.size());
    // print("values:", values);////////////////


    // softmaxで使う変数
    float maxPolicy = 0.0;

    for (auto [y, x, t] : moves) {
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


    // policys の最大が values[0] になるように調整
    for (auto [move, x] : policys) {
        policys[move] = values[0] * x / maxPolicy2;
    }


    // print("tmpPolicy.size()2:", tmpPolicy.size());//////////////
    // print("tmpPolicy2:", tmpPolicy);
    // print("values.size()2:", values.size());
    // print("values2:", values);////////////////


    for (auto [move, x] : policys) {
        float tmpUct = x + sqrt(2 * log(policys.size()));
        ucts.insert(make_tuple(tmpUct, 1, x, move));
    }


    /// TODO: 1.11111 が表示されるのはおかしい。
    vector<vector<float>> tmp(BOARDSIZE + 1, vector<float>(BOARDSIZE + 1, 1.11111));
    for (auto [move, x] : policys) {
        tmp[move.first][move.second] = x;
    }
    rep (i, 1, BOARDSIZE) {
        rep (j, 1, BOARDSIZE) {
            if (board[i + 1][j + 1] == 0) {
                cout << fixed << setprecision(4) << tmp[i][j] << " ";
            }
            else if (board[i + 1][j + 1] == 1) {
                cout << "###### ";
            } else if (board[i + 1][j + 1] == 2) {
                cout << "OOOOOO ";
            } else {
                cout << "?????? ";
            }
        }
        cout << endl;
    }
    print("pass:", tmp[0][0]);//////////////
    print("values:", values);
    cout << resetiosflags(ios_base::floatfield);


    return tie(teban, values[0], values[1], values[2]);
}


bool goBoard::UpdateUcts(tuple<int, float, float, float> input, pair<char, char> inputMove)
{
    auto [inputColor, inputWin, inputDraw, inputLose] = input;

    print("inputColor:", inputColor);////////////
    print("inputWin:", inputWin);
    print("inputDraw:", inputDraw);
    print("inputLose:", inputLose);
    print("teban:", teban);

    if (inputColor != teban) {
        swap(inputWin, inputLose);
    }

    set<tuple<double, int, float, pair<char, char>>> tmpUcts;

    ++numVisits;

    for (auto [uct, cnt, winSum, uctMove] : ucts) {
        if (inputMove == uctMove) {
            tmpUcts.insert(make_tuple(inputWin + sqrt(2 * log(numVisits) / cnt + 1), cnt + 1, winSum + inputWin, uctMove));
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

void goBoard::PrintBoard(char bit = 0b1)
{
    if (bit & 0b0001) {
        cout << (int)previousMove.first << " " << (int)previousMove.second << " " << (int)teban << endl;  ////////////////
        cout << "turn: " << (int)teban << endl;
        cout << "   1 2 3 4 5 6 7 8 9" << endl;
        rep (i, board.size()) {
            if (i == 0 || i == BOARDSIZE + 1) {
                continue;
            }
            cout << setw(2) << setfill(' ') << i << " ";
            rep (j, board.size()) {
                if (board[i][j] == 3) {
                }
                else if (board[i][j] == 1) {
                    cout << "● ";
                }
                else if (board[i][j] == 2) {
                    cout << "○ ";
                }
                else if (i == 1 && j == 1) {
                    cout << "┌ ";
                }
                else if (i == BOARDSIZE && j == 1) {
                    cout << "└ ";
                }
                else if (i == 1 && j == BOARDSIZE) {
                    cout << "┐ ";
                }
                else if (i == BOARDSIZE && j == BOARDSIZE) {
                    cout << "┘ ";
                }
                else if (i == 1) {
                    cout << "┬ ";
                }
                else if (i == BOARDSIZE) {
                    cout << "┴ ";
                }
                else if (j == 1) {
                    cout << "├ ";
                }
                else if (j == BOARDSIZE) {
                    cout << "┤ ";
                }
                else {
                    cout << "+ ";
                }
            }
            cout << endl;
        }
    }

    if (bit & 0b010) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cout << (int)board[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    if (bit & 0b100) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cout << setw(3) << setfill(' ') << idBoard[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    if (bit & 0b1000) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cout << setw(3) << setfill(' ') << libs[idBoard[i][j]] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    // ニューラルネットワーク入力用
    if (bit & 0b10000) {
        cout << (int)previousMove.first << " " << (int)previousMove.second << " " << (int)teban << endl;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cout << (int)board[i][j] << " ";
            }
            cout << endl;
        }
    }
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
    if (IsIllegalMove(y, x, color)) {///////////////
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
//     std::cout << data << std::endl;
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
                    cout << col << " ";
                }
                cout << endl;
            }
            cout << endl;
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

    goBoard* root(new goBoard(v));


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

        // cout << "numWin: " << numWin << endl;
        // cout << "numVisit: " << numVisit << endl;
        // cout << "root->numVisits: " << root->numVisits << endl;

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

    root->SucceedRoot(get<3>(ans));
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

        // cout << "numWin: " << numWin << endl;
        // cout << "numVisit: " << numVisit << endl;
        // cout << "root->numVisits: " << root->numVisits << endl;

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

    cout << rootPtr << endl;
    root->SucceedRoot(get<3>(ans));
    cout << rootPtr << endl;


    return 0;
}


int PutStoneCnt = 0;

int Test()
{
    samplesCommon::Args args;

    args.runInInt8 = false;
    args.runInFp16 = false;
    args.runInBf16 = false;

    TensorRTOnnxIgo tensorRT(initializeSampleParams(args, "test2.onnx"));

    tensorRT.build();

    auto saiki = [tensorRT](auto self, goBoard* ptr) -> tuple<int, float, float, float>
    {
        print("saiki", ptr->moveCnt);////////////////
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
            print("PutStoneCnt", ++PutStoneCnt);
            goBoard *nextPtr = ptr->PutStone(nextMove.first, nextMove.second, color);
                
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

    print(rootPtr->policys);
    print(rootPtr->values);

    float sum = 0.0;
    for (auto x : rootPtr->policys) {
        sum += x.second;
    }
    print("policyssum:", sum);

    sum = 0.0;
    for (auto x : rootPtr->values) {
        sum += x;
    }
    print("valuessum:", sum);

    print("ucts:", rootPtr->ucts);

    int saikiCnt = 0;
    rep (100) {
        print("saikiCnt:", saikiCnt++);////////////////
        saiki(saiki, rootPtr);
    }
    print("ucts:", rootPtr->ucts);

    pair<char, char> ans;
    int ansCnt = -1;
    for (auto x : rootPtr->ucts) {
        if (get<1>(x) >= ansCnt) {
            if (get<1>(x) == ansCnt && get<0>(x) < get<0>(ans)) {
                continue;
            }
            ansCnt = get<1>(x);
            ans = get<3>(x);
        }
    }

    print("ans:", ans);

    return 0;
}


int main()
{
    Test();
    // MonteCarloTreeSearch();
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