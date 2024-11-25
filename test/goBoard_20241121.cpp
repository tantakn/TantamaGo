// clang-format off
#define _GLIBCXX_DEBUG
#include <bits/stdc++.h>
using namespace std;
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
struct goBoard {
    vector<vector<char>> boardRaw;

    vector<vector<char>> board = boardRaw;

    set<vector<vector<char>>> boardHistory;

    /**
     * @brief
     *
     * @param input BOADSIZExBOARDSIZEの二次元配列
     * @return true
     * @return false
     */
    bool InputBoard(vector<vector<char>> input);

    /**
     * @brief
     *
     * @param opt intで表示するかどうか
     */
    void PrintBoard(string opt = "");

    /**
     * @brief 与えられた座標の石の呼吸点を数える
     *
     * @param x
     * @param y
     * @return int 呼吸点の数。-1なら石がない
     */
    int CountLiberties(int x, int y);

    /**
     * @brief 
     * 
     */
    bool IsLegal(int x, int y, char color);


    goBoard()
        : boardRaw(BOARDSIZE + 2, vector<char>(BOARDSIZE + 2, 0b0))
    {
        rep (i, BOARDSIZE + 2) {
            boardRaw[0][i] = 0b11;
            boardRaw[BOARDSIZE + 1][i] = 0b11;
            boardRaw[i][0] = 0b11;
            boardRaw[i][BOARDSIZE + 1] = 0b11;
        }
    }

    goBoard(vector<vector<char>> input)
        : boardRaw(BOARDSIZE + 2, vector<char>(BOARDSIZE + 2, 0b0)), board(input)
    {
        rep (i, BOARDSIZE + 2) {
            boardRaw[0][i] = 0b11;
            boardRaw[BOARDSIZE + 1][i] = 0b11;
            boardRaw[i][0] = 0b11;
            boardRaw[i][BOARDSIZE + 1] = 0b11;
        }
    }
};

bool goBoard::InputBoard(vector<vector<char>> input)
{
    if (int(input.size()) != BOARDSIZE) {
        return false;
    }
    for (auto row : input) {
        if (int(row.size()) != BOARDSIZE) {
            return false;
        }
    }
    vector<vector<char>> tmp = boardRaw;

    rep (i, 1, BOARDSIZE + 1) {
        rep (j, 1, BOARDSIZE + 1) {
            tmp[i][j] = input[i - 1][j - 1];
        }
    }

    board = tmp;
    return true;
};

void goBoard::PrintBoard(string opt)
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
        rep (i, board.size()) {
            rep (j, board.size()) {
                if (i == 0 || j == 0 || i == BOARDSIZE + 1 || j == BOARDSIZE + 1) {
                }
                else if ((board[i][j] & 0b11) == 0b01) {
                    cout << "● ";
                }
                else if ((board[i][j] & 0b11) == 0b10) {
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

int goBoard::CountLiberties(int x, int y)
{
    assert(x >= 1 && x <= BOARDSIZE && y >= 1 && y <= BOARDSIZE);

    if ((board[y][x] & 0b11) == 0b0) {
        return -1;
    };

    int color = board[y][x] & 0b11;

    vector<vector<char>> boardSearched = boardRaw;
    boardSearched[y][x] = 1;

    queue<pair<int, int>> bfs;

    vector<pair<int, int>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    int cnt = 0;

    for (auto dir : directions) {
        int nx = x + dir.first;
        int ny = y + dir.second;

        if (boardSearched[ny][nx]) {
            continue;
        }
        else if ((board[ny][nx] & 0b11) == 0b0) {
            boardSearched[ny][nx] = 1;
            ++cnt;
            // board[ny][nx] += 0b100;  /////////////
            continue;
        }
        else if ((board[ny][nx] & 0b11) == color) {
            bfs.push({nx, ny});
        }
    }

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
            else if ((board[ny][nx] & 0b11) == 0b0) {
                boardSearched[ny][nx] = 1;
                ++cnt;
                // board[ny][nx] += 0b100;  /////////////
                continue;
            }
            else if ((board[ny][nx] & 0b11) == color) {
                bfs.push({nx, ny});
            }
        }
    }
    return cnt;
};

bool goBoard::IsLegal(int w, int h, char color)
{
    assert(w >= 1 && w <= BOARDSIZE && h >= 1 && h <= BOARDSIZE);

    if (board[h][w] != 0b0) {
        return false;
    }
    
    vector<pair<int, int>> direction = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    for (auto dir : direction) {
        int nw = w + dir.first;
        int nh = h + dir.second;

        if (board[nh][nw] == color) {//TODO:意外と難しい？
        }
    }
    for (auto dir : direction) {
        int nw = w + dir.first;
        int nh = h + dir.second;

        if (CountLiberties(nw, nh) == 1) {//TODO:
        }
    }
    return true;
};


int main()
{
    goBoard board;
    board.InputBoard(
        // clang-format off
{{
1,1,0,0,0,0,0,0,0}, {
1,1,0,0,0,0,0,0,0}, {
0,0,0,1,1,1,0,0,0}, {
0,0,0,1,0,1,0,0,0}, {
0,0,0,1,0,0,0,0,0}, {
0,0,0,0,0,0,0,0,0}, {
0,0,0,0,0,0,0,0,0}, {
1,0,0,0,0,2,2,0,0}, {
2,1,0,0,0,2,2,0,0}}
        // clang-format on
    );


    board.PrintBoard("int");
    board.PrintBoard();

    int x, y;
    cin >> x >> y;

    print("Liberties:", board.CountLiberties(x, y));
    print("color:", (int)board.board[y][x]);


    board.PrintBoard();



    return 0;
}




// // clang-format off
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