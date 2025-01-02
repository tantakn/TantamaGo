#define goBoard_cpp_INCLUDED

#ifndef goBoard_hpp_INCLUDED
#include "goBoard.hpp"
#define goBoard_hpp_INCLUDED
#endif

#define dbg_flag
#ifdef dbg_flag
ll g_node_cnt = 0;
#endif


mt19937 mt(random_device{}());


goBoard::goBoard()
    : board(rawBoard), idBoard(rawIdBoard), teban(1), parent(nullptr)
{
    libs[-1] = INF;
}

goBoard::goBoard(vector<vector<char>> inputBoard)
    : board(InputBoardFromVec(inputBoard)), idBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0)), teban(1)
{
    /// TODO: teban の扱いを考える
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
    : parent(&inputparent), board(inputparent.board), idBoard(inputparent.idBoard), libs(inputparent.libs), stringIdCnt(inputparent.stringIdCnt), history(inputparent.history), teban(1), previousMove(make_pair(y, x))
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE && (putcolor == 0b01 || putcolor == 0b10) || putcolor == 0);

#ifdef dbg_flag
    g_node_cnt++;
#endif

    if (putcolor == 0) {
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

    // PrintBoard(1);
}

goBoard::~goBoard()
{
    for (auto m : childrens) {
        goBoard* child = m.second;
        if (debugFlag & 0b100) print("delete child");
        delete child;
        child = nullptr;
    }
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
    // if (opt == "int") {
    //     rep (i, 1, board.size() - 1) {
    //         rep (j, 1, board.size() - 1) {
    //             cout << (int)board[i][j] << " ";
    //         }
    //         cout << endl;
    //     }
    // }
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

int goBoard::IsLegalMove(int y, int x, char color)
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE && (color == 0b01 || color == 0b10) || (x == 0 && y == 0 && color == 0));

    if (isEnded) {
        return 5;
    }

    if (color == 0) {
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
        }
    }
    if (isFillEye) {
        return 4;
    }

    return 0;
};

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
    assert(!IsLegalMove(y, x, color));

    // auto ptr = make_unique<goBoard>(*this, y, x, color);
    // goBoard* newBoardPtr = newBoard.get();
    goBoard* p = new goBoard(*this, y, x, color);
    childrens[make_pair(y, x)] = p;

    if (debugFlag & 0b10) p->PrintBoard();

    return p;


    // goBoard* newBoard = new goBoard(*this, y, x, color);
    // childrens.push_back(newBoard);
    // // unique_ptr<goBoard> ptr = make_unique<goBoard>(*this, y, x, color);
    // // std::shared_ptr<goBoard> child = std::make_shared<goBoard>(*this);

    // childrens.push_back(newBoard);
    // return *newBoard;
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
        if (!IsLegalMove(y, x, teban)) {
            return {y, x, teban};
        }
        else if (debugFlag & 0b100) {
            print(x, y, IsLegalMove(y, x, teban));
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

    print("blackScore:", blackScore);
    print("whiteScore:", whiteScore);
    print("komi:", komi);

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

vector<vector<vector<double>>> goBoard::MakeInputPlane()
{
    vector<vector<vector<double>>> inputPlane(6, vector<vector<double>>(BOARDSIZE, vector<double>(BOARDSIZE, 0)));

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



int main()
{
    ofstream ofs("planejson.txt");

    goBoard* testboard(new goBoard());

    testboard->PrintBoard();

    while (true) {
        auto [y, x, z] = testboard->GenRandomMove();
        if (y == -1) {
            testboard->PrintBoard(0b111111);
            print(testboard->CountResult());
            break;
        }
        nlohmann::json j(testboard->MakeInputPlane());
        cout << j.dump() << endl;
        testboard = testboard->PutStone(y, x, z);
        print(testboard->ToJson());
    }

    print(testboard->ToJson());


    print(g_node_cnt);


    //     unique_ptr<goBoard> testboard(new goBoard(
    //         // clang-format off
    // {{
    // 0,1,0,0,0,0,2,1,0}, {
    // 1,1,0,0,0,0,0,2,2}, {
    // 0,1,0,1,1,1,0,0,2}, {
    // 2,1,0,1,0,1,0,2,0}, {
    // 1,1,0,1,1,0,0,0,2}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,2,0,0,0}, {
    // 1,0,0,0,1,0,2,0,0}, {
    // 0,1,0,0,0,2,2,0,0}}
    //         // clang-format on
    //         ));

    //     testboard->PrintBoard();

    //     goBoard* ptr = testboard->PutStone(8, 2, 0);
    //     while (true) {
    //         int x, y, z;
    //         cout << "x: ";
    //         cin >> x;
    //         if (x == -1) {
    //             ptr = nullptr;
    //             break;
    //         }
    //         else if (x == -2) {
    //             ptr->PrintBoard(0b111111);
    //             continue;
    //         }
    //         else if (x == -3) {
    //             auto[b, a, c] = ptr->GenRandomMove();
    //             x = a; y = b; z = c;
    //             print(x,y,z);
    //         } else {
    //             cout << "y: ";
    //             cin >> y;
    //             cout << "color: ";
    //             cin >> z;
    //         }
    //         if (x < 1 || x > BOARDSIZE || y < 1 || y > BOARDSIZE || (z != 1 && z != 2 && z != 0)) {
    //             print("Illegal Input");
    //             continue;
    //         }
    //         if (ptr->IsLegalMove(y, x, z)) {
    //             print("Illegal Move");
    //             continue;
    //         }
    //         goBoard* ptr2 = ptr->PutStone(y, x, z);
    //         ptr2->PrintBoard();
    //         ptr = ptr2;
    //     }


    //     goBoard board(
    //         // clang-format off
    // {{
    // 0,1,0,0,0,0,2,1,0}, {
    // 1,1,0,0,0,0,0,2,2}, {
    // 0,1,0,1,1,1,0,0,2}, {
    // 2,1,0,1,0,1,0,2,0}, {
    // 1,1,0,1,1,0,0,0,2}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,2,0,0,0}, {
    // 1,0,0,0,1,0,2,0,0}, {
    // 0,1,0,0,0,2,2,0,0}}
    //         // clang-format on
    //     );


    //     board.PrintBoard();
    //     board.DbgPrint();

    // while (true) {
    //     int x, y, z;
    //     cout << "x: ";
    //     cin >> x;
    //     if (x == -1) {
    //         break;
    //     }else if (x == -2) {
    //         board.DbgPrint();
    //         continue;
    //     }
    //     cout << "y: ";
    //     cin >> y;
    //     cout << "color: ";
    //     cin >> z;


    //     print("Liberties:", board.CountLiberties(y, x));
    //     print("color:", (int)board.board[y][x]);
    //     print("IsLegalMove:", board.IsLegalMove(y, x, z));

    //     board.PrintBoard();
    // }

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