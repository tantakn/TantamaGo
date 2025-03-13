/// TODO: 同じ形のノードを併合する。デストラクタを工夫する必要がある？
/// TODO: 回転・反転させて同じ形のを併合する。添字の順で、黒・白の優先順でなるべく石が多くなるようにする？
/// TODO: マルチスレッドをやってみる。関数化すれば意外と簡単？
/// TODO: avxを使ってみる。19路だと難しい？




const vector<pair<char, char>> directions = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

/// @brief 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁
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

/// @brief 0は空点、-1は壁
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
 * @brief グローバル変数 goBoard* rootPtr = nullptr; が必要。
 * 
 */
struct goBoard {
    /// @brief 現在の手番。1: 黒, 2: 白
    char teban;


    /// @brief 手数（≒盤上の石の数）
    int moveCnt;


    /// @brief [y, x] 直前の着手。[-1, -1] は初期状態、[0, 0] はパス。
    pair<char, char> previousMove = make_pair(-1, -1);


    /// @brief 終局図かどうか
    bool isEnded = false;


    /// @brief ルートノードかどうか
    bool isRoot;


    // /// @brief ExpandNode() されたかどうか
    // bool isNotExpanded = true;


    /// @brief 0b00: 空点, 0b01: 黒, 0b10: 白, 0b11: 壁。壁を含み、要素数 (BOARDSIZE + 2) * (BOARDSIZE + 2)。つまり、端は board[BOARDSIZE + 1][BOARDSIZE + 1]。
    vector<vector<char>> board;


    /// @brief 連のid。0は空点、-1は壁
    vector<vector<int>> idBoard;


    /// @brief <連のid, 連の呼吸点の数>。壁はINF
    map<int, int> libs;


    /// @brief 超コウルール用の履歴。手番の情報はない。
    set<vector<vector<char>>> history;


    /// @brief idBoardのstringのidのカウント
    int stringIdCnt = 1;


    /// @brief 親盤面
    /// TODO: = nullptr でいい？
    goBoard *parent;


    /// @brief 子盤面
    map<pair<char, char>, goBoard *> childrens;


    /// @brief 推論の結果にsoftmaxを適用したもののうちの合法手のみのマップ。座標は盤外あり。y == 0 && x == 0 がパス。policy値はvalue値から計算された勝率。
    map<pair<char, char>, float> policys;


    /// @brief 推論の結果にsoftmaxを適用したもの、多分 [相手の手番の勝率（例：初期局面なら白の勝率）, 引き分けの確率, 現在の勝率]
    vector<float> values;

    mutex uctsMutex;
    // mutex uctsMutex;

    /// @brief <uct, この手の探索回数, この手の勝率の合計, 着手>。着手は piar<0, 0> でパス。rbegin(ptr->ucts) みたく使う。
    /// uct = この手の勝率の合計 / この手の探索回数 + sqrt(2 * log(現局面の総探索回数) / この手の探索回数)
    set<tuple<double, int, float, pair<char, char>>> ucts;
    // /// @brief tuple<uct, この手の探索回数, この手の勝利回数, 着手>
    // /// uct = この手の勝利回数 / この手の探索回数 + sqrt(2 * log(現局面の総探索回数) / この手の探索回数)
    // set<tuple<double, int, int, pair<char, char>>> ucts;


    /// @brief <puct, この手の探索回数, この手のvalueの合計, 着手>。着手は piar<0, 0> でパス。rbegin(ptr->pucts) みたく使う。
    /// 授業で教わった定義：Valueの平均値 + PUCT_C * sqrt(log(この手の探索回数)) / (1 + 現局面の総探索回数)
    /// ネットで見つけた定義：puct = (log((1 + この手の探索回数 + PUCT_C_BASE) / PUCT_C_BASE) + PUCT_C_INIT) * この手の勝率の合計 / この手の探索回数 + sqrt(2 * log(現局面の総探索回数) / この手の探索回数)
    set<tuple<double, int, float, pair<char, char>>> pucts;


    /// @brief 子ノードの探索回数の合計
    int numVisits = 0;


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
     * @brief rootPtr を指定した手の盤面にして、自身と他の子孫を削除する。
     * 
     * @param move 
     * @return goBoard* 
     */
    goBoard* SucceedRoot(goBoard*& rootPtr, pair<char, char> move);


    /**
     * @brief debugFlag & 1<<31 & 1<<29 で推論の結果を表示する。
     * 
     * @return tuple<int, float, float, float> color、colorが負ける確率、引き分けの確率、colortが勝つ確率
     */
    tuple<int, float, float, float> ExpandNode(pair<vector<float>, vector<float>> input);


    /**
     * @brief 
     * 
     * @param tensorRT 
     * @return tuple<int, float, float, float> color、colortが勝つ確率、引き分けの確率、colorが負ける確率
     */
    tuple<int, float, float, float> ExpandNode(TensorRTOnnxIgo tensorRT);


    /**
     * @brief ucts を更新する。
     * 
     */
    bool UpdateUcts(tuple<int, float, float, float> input, pair<char, char> move);


    /**
     * @brief Get the Ans object
     * 
     * @return pair<char, char> move(y, x)
     */
    pair<char, char> GetBestMove();


    /**
     * @brief goBoard::goBoard(vector<vector<char>> inputBoard) で使う。内蔵すべき？
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
     * @param opt "int"を指定すると内部の数値を表示する。
     */
    void PrintBoard(ll bit);


    string ToJson();


    /**
     * @brief 呼吸点の数を数える。壁はINFを返す
     *
     * @param y
     * @param x
     * @param board 現在の盤面。空ならthis->boardを使う。
     * @return int
     */
    int CountLiberties(int y, int x, vector<vector<char>> board);


    bool IsBestMoveCrucial();


    /**
     * @brief y == 0 && x == 0 でパスのはず。
     *
     * @param y
     * @param x
     * @param color
     * @return true
     * @return false
     */
    int IsIllegalMove(int y, int x, char color);


    /**
     * @brief debugFlag & 1 << 30 で合法手を表示する。
     * 
     * @return vector<tuple<char, char, char>> : <y, x, teban>。
     * 盤外あり。
     * y == 0 && x == 0 でパス。
     */
    vector<tuple<char, char, char>> GenAllLegalMoves();


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
     * @brief 指定の場所に石を置いた子ノードをnewして、childrens に追加する。
     *
     * @param y
     * @param x
     * @param color
     * @return goBoard
     */
    goBoard* PutStone(int y, int x, char color);


    /**
     * @brief とりあえず、パス以外の合法手があればパス以外を選択。
     * 
     * @return tuple<char, char, char> y, x, teban
     */
    tuple<char, char, char> GenRandomMove();


    /**
     * @brief 暫定的に勝敗を返してる。
     * 
     * @return double 黒地 - 白地 - コミ
     */
    double CountResult(bool dbg);


    bool TestPipe();


    vector<vector<vector<float>>> MakeInputPlane();


    goBoard();


    goBoard(goBoard &inputparent, int y, int x, char putcolor);


    /// TODO: parent と children の処理を書く
    goBoard(vector<vector<char>> inputBoard, char inputTeban);


    ~goBoard();
};



/// @brief GPT では縦軸の値を英字の abcdefghjklmnopqrst (iがない) で表すため、char型の文字をint型の数字に変換する
/// @param s 
/// @return 変換後の数値
int ConvertChar(char s);


/// @brief GPT では縦軸の値を英字の abcdefghjklmnopqrst (iがない) で表すため、int型の数字をchar型の文字に変換する
/// @param n 
/// @return 変換後の文字
char ConvertInt(int n);



string Gpt(const string input, goBoard*& rootPtr, TensorRTOnnxIgo& tensorRT, thread& searchThread, int thinkTime, bool ponder);

void SearchLoop(goBoard* rootPtr, TensorRTOnnxIgo& tensorRT);


