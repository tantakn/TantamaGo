#ifndef myMacro_INCLUDED
#include "myMacro.hpp"
#define myMacro_INCLUDED
#endif


constexpr int debugFlag = 0b11;

constexpr int BOARDSIZE = 9;

constexpr double komi = 7.5;

const vector<pair<char, char>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

vector<vector<char>> rawBoard = []()
{
    vector<vector<char>> tmpBoard(BOARDSIZE + 2, vector<char>(BOARDSIZE + 2, 0b0));
    rep (i, BOARDSIZE + 2) {
        rawBoard[0][i] = 0b11;
        rawBoard[BOARDSIZE + 1][i] = 0b11;
        rawBoard[i][0] = 0b11;
        rawBoard[i][BOARDSIZE + 1] = 0b11;
    }
    return tmpBoard;
}();

vector<vector<int>> rawIdBoard = []() 
{
    vector<vector<int>> tmpIdBoard(BOARDSIZE + 2, vector<int>(BOARDSIZE + 2, 0));
    rep (i, BOARDSIZE + 2) {
        rawIdBoard[0][i] = -1;
        rawIdBoard[BOARDSIZE + 1][i] = -1;
        rawIdBoard[i][0] = -1;
        rawIdBoard[i][BOARDSIZE + 1] = -1;
    }
    return tmpIdBoard;
}();

/**
 * @brief
 *
 */
struct goBoard {
    /// @brief 現在の手番。0b01: 黒, 0b10: 白
    char color;

    /// @brief 直前の手がパスかどうか
    bool isPreviousPass = false;

    /// @brief 終局図かどうか
    bool isEnded = false;

    /// 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁。
    vector<vector<char>> board;

    /// @brief 連のid。0は空点、-1は壁
    vector<vector<int>> idBoard;

    /// @brief 連の呼吸点の数。壁はINF
    map<int, int> libs;

    /// @brief 超コウルール用の履歴
    set<vector<vector<char>>> history;

    /// @brief idBoardのstringのidのカウント
    int stringIdCnt = 1;

    /// @brief 親盤面
    /// TODO: = nullptr でいい？
    goBoard *parent;

    /// @brief 子盤面
    map<tuple<char, char>, goBoard *> childrens;

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
    void PrintBoard(char bit);

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
    int IsLegalMove(int y, int x, char color);

    /**
     * @brief idBoard[y][x] == 0 な石とつながっている石で新しい連を作る
     *
     * @param y
     * @param x
     */
    void ApplyString(int y, int x);

    /**
     * @brief
     *
     * @param y
     * @param x
     */
    void DeleteString(int y, int x);

    /**
     * @brief
     *
     * @param y
     * @param x
     * @param color
     * @return goBoard
     */
    goBoard* PutStone(int y, int x, char color);

    tuple<char, char, char> GenRandomMove();

    double CountResult();

    goBoard();

    goBoard(goBoard &inputparent, int y, int x, char putcolor);

    /// TODO: 引数なしの初期化関数を作る
    /// TODO: parent と children の処理を書く
    goBoard(vector<vector<char>> inputBoard);

    ~goBoard();
};
