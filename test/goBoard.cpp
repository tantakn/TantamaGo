// clang-format off
#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;

const int INF = 1047483647;
using ll = int_fast64_t;
using ull = uint_fast64_t;
#define myall(x) (x).begin(), (x).end()
#define MyWatch(x) (double)(clock() - (x)) / CLOCKS_PER_SEC
#define istrue ? assert(true) : assert(false && "istrue")
#define _rep0(a) for (uint_fast64_t _tmp_i = 0; _tmp_i < UINT_FAST64_MAX; ++_tmp_i, assert(_tmp_i < INT_MAX))
#define _rep1(a) for (int_fast64_t _tmp_i = 0; _tmp_i < (int_fast64_t)(a); ++_tmp_i)
#define _rep2(i, a) for (int_fast64_t i = 0; i < (int_fast64_t)(a); ++i)
#define _rep3(i, a, b) for (int_fast64_t i = (int_fast64_t)(a); i < (int_fast64_t)(b); ++i)
#define _print0(a) cout << endl
#define _print1(a) cout << (a) << endl
#define _print2(a, b) cout << (a) << ", " << (b) << endl
#define _print3(a, b, c) cout << (a) << ", " << (b) << ", " << (c) << endl
#define _overload(a, b, c, d, e ...) d
#define rep(...) _overload(__VA_ARGS__ __VA_OPT__(,)  _rep3, _rep2, _rep1, _rep0)(__VA_ARGS__)
#define print(...) _overload(__VA_ARGS__ __VA_OPT__(,)  _print3, _print2, _print1, _print0)(__VA_ARGS__)

template<class T>bool chmax(T &a, const T &b) { if (a<b) { a=b; return 1; } return 0; }
template<class T>bool chmin(T &a, const T &b) { if (b<a) { a=b; return 1; } return 0; }
// clang-format on


int BOARDSIZE = 9;

vector<pair<char, char>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

vector<vector<char>> boardRaw = []()
{
    vector<vector<char>> tmpBoard(BOARDSIZE + 2, vector<char>(BOARDSIZE + 2, 0b0));
    rep (i, BOARDSIZE + 2) {
        boardRaw[0][i] = 0b11;
        boardRaw[BOARDSIZE + 1][i] = 0b11;
        boardRaw[i][0] = 0b11;
        boardRaw[i][BOARDSIZE + 1] = 0b11;
    }
    return tmpBoard;
}();

/**
 * @brief 
 * 
 */
struct goBoard {
    /// 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁。
    vector<vector<char>> board;

    /// @brief stringのid。0は空点、-1は壁
    vector<vector<int>> idBoard;

    /// @brief stringのliberty。0は空点、INFは壁
    vector<vector<int>> libBoard;

    /// @brief 超コウルール用の履歴
    set<vector<vector<char>>> history;

    /// @brief stringのidのカウント
    int stringIdCnt = 1;

    /// @brief 親盤面
    /// TODO: = nullptr でいい？
    goBoard* parent = nullptr;

    /// @brief 子盤面
    vector<goBoard*> children;

    /// @brief 
    vector<vector<double>> policyBoard;

    /// @brief 
    double value;

    // /**
    //  * @brief なんかうまくいかない
    //  *
    //  * @param row
    //  * @param col
    //  * @return const char&
    //  */
    // const char &operator()(int row, int col)
    // {
    //     return board[col][row] & 0b11;
    // }

    /**
     * @brief
     *
     * @param input BOARDSIZExBARDSIZEの盤面の配列。下に示すような形式で入力する
     * @return vector<vector<char>> 盤外（壁）を含めた盤面の配列
     */
    vector<vector<char>> InputBoardFromVec(vector<vector<char>> input);
    //     // clang-format off
    // {{
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}, {
    // 0,0,0,0,0,0,0,0,0}};
    // // clang-format on

    /**
     * @brief 盤面をいい感じに表示する
     *
     * @param opt "int"を指定すると内部の数値を表示する。DbgPrint() もあるよ
     */
    void PrintBoard(string opt);

    /**
     * @brief 呼吸点の数を数える。壁はINFを返す
     *
     * @param y
     * @param x
     * @return int
     */
    int CountLiberties(int y, int x);

    /**
     * @brief
     *
     * @param y
     * @param x
     * @param color
     * @return true
     * @return false
     */
    bool IsLegalMove(int y, int x, char color);

    /**
     * @brief
     *
     * @param bit フラグ。0b001: 盤面, 0b010: idBoard, 0b100: libBoard
     */
    void DbgPrint(char bit);

    /**
     * @brief
     *
     * @param y
     * @param x
     */
    void MakeString(int y, int x);

    /**
     * @brief 
     * 
     * @param y 
     * @param x 
     * @param color 
     * @return goBoard 
     */
    goBoard PutStone(int y, int x, char color);

    goBoard(goBoard *parent)
        : parent(parent), board(parent->board), idBoard(parent->idBoard), libBoard(parent->libBoard), stringIdCnt(parent->stringIdCnt), history(parent->history) {
        parent->children.push_back(this);
        
    };

    /// TODO: 引数なしの初期化関数を作る
    /// TODO: parent と children の処理を書く
    goBoard(vector<vector<char>> inputBoard)
        : board(InputBoardFromVec(inputBoard)), idBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0)), libBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0))
    {
        rep (i, BOARDSIZE + 2) {
            idBoard[0][i] = -1;
            idBoard[BOARDSIZE + 1][i] = -1;
            idBoard[i][0] = -1;
            idBoard[i][BOARDSIZE + 1] = -1;

            libBoard[0][i] = INF;
            libBoard[BOARDSIZE + 1][i] = INF;
            libBoard[i][0] = INF;
            libBoard[i][BOARDSIZE + 1] = INF;
        }

        int cnt = 1;
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (idBoard[i][j] == 0 && board[i][j] != 0) {
                    MakeString(i, j);
                }
            }
        }
    }
};


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

    vector<vector<char>> tmp = boardRaw;

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            tmp[i][j] = input[i - 1][j - 1];
        }
    }

    return tmp;
};

void goBoard::PrintBoard(string opt = "")
{
    if (opt == "int") {
        rep (i, 1, board.size() - 1) {
            rep (j, 1, board.size() - 1) {
                cout << (int)board[i][j] << " ";
            }
            cout << endl;
        }
    }
    else {
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
};

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

    vector<vector<char>> boardSearched = boardRaw;
    boardSearched[y][x] = 1;

    queue<pair<int, int>> bfs;

    bfs.push({x, y});

    int cnt = 0;

    while (!bfs.empty()) {
        auto [x, y] = bfs.front();
        bfs.pop();
        boardSearched[y][x] = 1;
        // print(x, y);  ////////////////

        for (auto dir : directions) {
            int nx = x + dir.first;
            int ny = y + dir.second;

            if (boardSearched[ny][nx]) {
                continue;
            }
            else if (board[ny][nx] == 0) {
                boardSearched[ny][nx] = 1;
                ++cnt;
                // board[ny][nx] += 0b100;  /////////////
                continue;
            }
            else if (board[ny][nx] == color) {
                bfs.push({nx, ny});
            }
        }
    }
    return cnt;
};

bool goBoard::IsLegalMove(int y, int x, char color)
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE && (color == 0b01 || color == 0b10));

    vector<vector<char>> boardTotta;

    // すでに石がある場合はfalse
    if (board[y][x] != 0) {
        return false;
    }

    // 自殺手 || 超コウルール で合法手でない

    // 打つ場所の4方が囲まれている && 囲っている自分の石の呼吸点がすべて1 && !囲っている相手の石に呼吸点が1のものがある なら自殺手

    // 超コウルールは取れる石を取ってから判定する

    bool isSuicide = true;
    set<int> takeEnemyIds;
    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (board[ny][nx] == 0) {
            isSuicide = false;
        }
        else if (board[ny][nx] == color) {
            if (libBoard[ny][nx] != 1) {
                isSuicide = false;
            }
        }
        else if (board[ny][nx] == 3 - color) {
            if (libBoard[ny][nx] == 1) {
                isSuicide = false;
                takeEnemyIds.insert(idBoard[ny][nx]);
            }
        }
    }
    if (isSuicide) {
        return false;
    }

    // 超コウルール
    // TODO: 以下動くか分からない
    vector<vector<char>> tmp = board;

    for (auto id : takeEnemyIds) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                if (idBoard[i][j] == id) {
                    tmp[i][j] = 0;
                }
            }
        }
    }
    if (history.count(tmp)) {
        return false;
    }

    return true;
};

void goBoard::MakeString(int y, int x)
{
    assert(x >= 1 && x <= BOARDSIZE + 1 && y >= 1 && y <= BOARDSIZE + 1);
    assert(idBoard[y][x] == 0);
    assert(board[y][x] != 0);

    ++stringIdCnt;

    char color = board[y][x];

    /// TODO: 2つの string をつなげるとき、それぞれの lib - 1 を足し合わせればいい？
    /// TODO: lib == 0 の string で assert 出したい
    int lib = CountLiberties(y, x); 

    queue<pair<int, int>> bfs;

    bfs.push({x, y});

    while (!bfs.empty()) {
        auto [x, y] = bfs.front();
        bfs.pop();

        idBoard[y][x] = stringIdCnt;
        libBoard[y][x] = lib;

        for (auto dir : directions) {
            char nx = x + dir.first;
            char ny = y + dir.second;

            if (board[ny][nx] == color && idBoard[ny][nx] != stringIdCnt) {
                bfs.push({nx, ny});
            }
        }
    }
}

void goBoard::DbgPrint(char bit = 0b111)
{
    if (bit & 0b001) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cout << (int)board[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    if (bit & 0b010) {
        rep (i, BOARDSIZE + 2) {
            rep (j, BOARDSIZE + 2) {
                cout << setw(3) << setfill(' ') << idBoard[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    if (bit & 0b100) {
        rep (i, 1, BOARDSIZE + 1) {
            rep (j, 1, BOARDSIZE + 1) {
                cout << setw(3) << setfill(' ') << libBoard[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

goBoard goBoard::PutStone(int y, int x, char color)
{
    /// TODO: parent とか children, history, libBoard とかの処理も書く。goBoard はポインタで返したほうがいい？
    /// TODO: goBoard はポインタで返したほうがいい？
    /// TODO: string の処理を考える

    assert(IsLegalMove(y, x, color));

    unique_ptr<goBoard> ptr = make_unique<goBoard>(*this);
    // std::shared_ptr<goBoard> child = std::make_shared<goBoard>(*this);
};


int main()
{
    unique_ptr<goBoard> board(new goBoard(
        // clang-format off
{{
0,1,0,0,0,0,2,1,0}, {
1,1,0,0,0,0,0,2,2}, {
0,1,0,1,1,1,0,0,2}, {
2,1,0,1,0,1,0,2,0}, {
1,1,0,1,1,0,0,0,2}, {
0,0,0,0,0,0,0,0,0}, {
0,0,0,0,0,2,0,0,0}, {
1,0,0,0,1,0,2,0,0}, {
0,1,0,0,0,2,2,0,0}}
        // clang-format on
    ));
    
    board->PrintBoard();



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