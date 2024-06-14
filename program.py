"""プログラムの情報。
"""
PROGRAM_NAME="TantamaGo"
PROTOCOL_VERSION="2"

# Version 0.0.0 : ランダムプレイヤの実装。
# Version 0.1.0 : SGFファイルの読み込み処理の実装。load_sgfコマンドの対応。
#                 配石パターンのデータ構造の追加。眼の判定、上下左右の空点判定等改善。
#                 着手履歴、Zobrist Hash、超劫の判定の実装。
# Version 0.2.0 : ニューラルネットワークの教師あり学習の実装。
#                 Policy Networkを使用した着手生成ロジックの実装。
# Version 0.2.1 : Residual Blockの構造を修正。学習の再実行。
# Version 0.3.0 : モンテカルロ木探索の実装。
# Version 0.3.1 : モンテカルロ木探索のValue更新処理のバグ修正。komi, get_komiコマンドのサポート。
# Version 0.4.0 : Sequential Halving Applied to Trees (SHOT) の実装。
# Version 0.5.0 : 探索時間の制御、time_left、time_settingsコマンドのサポート。
# Version 0.6.0 : Gumbel AlphaZero方式の強化学習の実装。ネットワークの構造改善。
# Version 0.6.1 : --batch-sizeオプションの追加。
# Version 0.6.2 : 2回連続パスした時にノードを展開しないように変更。
# Version 0.6.3 : 探索回数を増やした時に強くならないバグの修正。time_leftコマンドのバグ修正。
# Version 0.6.4 : time_leftコマンドが来ないと持ち時間を正しく消費しないバグの修正。
# Version 0.6.5 : GPU使用時のGoGUI解析コマンドが落ちるバグの修正。--window-sizeオプションのバグ修正。
#                 思考時間管理処理の改良。
# Version 0.7.0 : lz-analyze, lz-genmove_analyze, cgos-analyze, cgos-genmove_analyzeコマンドのサポート。
#                 強化学習に関するバグと超劫の判定処理のバグの修正。
# Version 0.7.1 : 解析コマンドのバグ修正。
# Version 0.8.0 : SHOTでMixed value approximationを使うように変更
#                 持ち時間の残りが少なくなった時にプログラムが落ちる不具合を修正。
#                 強化学習の棋譜生成時に経過情報の表示を追加。
# Version 0.9.0 : undo, fixed_handicapコマンド、コマンドID付きGTPコマンドのサポート。
#                 不正なGTPコマンドの応答誤りを修正。
# Version 0.9.1 : PVに空文字が混入する不具合の修正。
# Version 0.9.2 : list_commandsにundoコマンドが漏れている不具合を修正。
VERSION="0.9.2"
