"""学習データの生成処理。
"""
import glob
import os
import random
import numpy as np
from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes, generate_target_data, generate_rl_target_data
from sgf.reader import SGFReader
from learning_param import BATCH_SIZE, DATA_SET_SIZE
from typing import List
import click


import time, datetime################


def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray, value_data: np.ndarray, kifu_counter: int) -> None:
    """学習データをnpzファイルとして出力する。引数のnp配列それぞれの辞書として保存する。

    Args:
        save_file_path (str): 保存するファイルパス。
        input_data (np.ndarray): 入力データ。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        kifu_counter (int): データセットにある棋譜データの個数。
    """

    # 辞書化して保存
    save_data = {
        "input": np.array(input_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "kifu_count": np.array(kifu_counter)
    }

    # killed って出るときはメモリ足りてない
    np.savez_compressed(save_file_path, **save_data)


# pylint: disable=R0914
def generate_supervised_learning_data(program_dir: str=None, kifu_dir: str=None, board_size: int=9) -> None:
    """教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。{kifu_dir}/*.sgf。* は 1 始まり。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
    """
    assert kifu_dir is not None, "kifu_dir is None."
    assert program_dir is not None, "program_dir is None."


    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_data start")####################
    print(f"    BATCH_SIZE: {BATCH_SIZE}")
    print(f"    DATA_SET_SIZE: {DATA_SET_SIZE}")
    kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
    print(f"    kifu_num: {kifu_num}")#############


    dt_watch = datetime.datetime.now()################

    board = GoBoard(board_size=board_size)

    input_data = []
    """入力データ。説明変数たち。"""

    policy_data = []
    """moveのデータ。目的変数（ターゲットデータ）たち。"""

    value_data = []
    """勝敗のデータ。目的変数？たち。"""

    kifu_counter = 1
    """npzファイルに書き込む棋譜データの個数を数えておく。npzにも書き込む。"""

    data_counter = 0
    """f"data/sl_data_{data_counter}"""

    cnt = 0###############
    """デバグ用。これまでの棋譜の総数"""

    # 局のループ
    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        cnt += 1###############
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""

        # 手のループ
        for pos in sgf.get_moves():
            # 対称形でかさ増し
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)

            # 手を一手進める
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            # input_data の局面の手番が勝者の場合は 2 にする。
            value_label = 2 - value_label

        # データセットのサイズを超えたら保存
        if len(value_data) >= DATA_SET_SIZE:
            # データを保存
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)

            # 保存したデータを削除
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]

            kifu_counter = 1
            data_counter += 1


            print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_npz")#####################
            print(f"    saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})")
            print(f"    cnt: {cnt} / {kifu_num}kyoku")
            dt_watch = datetime.datetime.now()


        kifu_counter += 1


    # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)




def generate_reinforcement_learning_data(program_dir: str, kifu_dir_list: List[str], board_size: int=9) -> None:
    """強化学習で使用するデータを生成し、保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリ。
        kifu_dir_list (List[str]): 棋譜ファイルを保存しているディレクトリパスのリスト。
        board_size (int, optional): 碁盤の大きさ。デフォルトは9。
    """
    dt_watch = datetime.datetime.now()#############
    print(f"🐾generate_reinforcement_learning_data {dt_watch}🐾")##################

    board = GoBoard(board_size=board_size)

    input_data = []
    policy_data = []
    value_data = []

    kifu_counter = 1
    data_counter = 0

    kifu_list = []
    for kifu_dir in kifu_dir_list:
        # kifu_list.extend(glob.glob(os.path.join(kifu_dir, "*", "*.sgf"))) # natukazeの棋譜のフォルダ形式に合うようにしたもの
        kifu_list.extend(glob.glob(os.path.join(kifu_dir, "*.sgf")))
    random.shuffle(kifu_list)

    for kifu_path in kifu_list:
        board.clear()
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        target_index = sorted(np.random.permutation(np.arange(sgf.get_n_moves()))[:8])
        """総手数以下の数からランダムで８個選んで選んだのをソート"""
        sym_index_list = np.random.permutation(np.arange(8))
        sym_index = 0
        #target_index = np.random.permutation(np.arange(sgf.get_n_moves()))[:1]
        #sym = np.random.permutation(np.arange(8))[0]
        for i, pos in enumerate(sgf.get_moves()):
            if i in target_index:
                sym = sym_index_list[sym_index]
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_rl_target_data(board, sgf.get_comment(i), sym))
                value_data.append(value_label)
                sym_index += 1
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            value_label = 2 - value_label

        if len(value_data) >= DATA_SET_SIZE:
            _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]
            kifu_counter = 1
            data_counter += 1

        kifu_counter += 1

    # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)

if __name__ == "__main__":
    # pylint: disable=R0914
    @click.command()
    @click.option('--program-dir', type=click.STRING, \
        help="プログラムのホームディレクトリのパス。")
    @click.option('--kifu-dir', type=click.STRING, \
        help="SGFファイルを格納しているディレクトリのパス。")
    @click.option('--board_size', type=click.INT, \
        help="碁盤のサイズ. Defaults to 9.")
    def tmp_generate_supervised_learning_data(program_dir: str=None, kifu_dir: str=None, board_size: int=9) -> None:
        generate_supervised_learning_data(os.path.dirname(__file__), kifu_dir, board_size)













def generate_supervised_learning_worker(program_dir: str, kifu_list: list, board_size: int=9) -> None:

    
    board = GoBoard(board_size=board_size)

    input_data = []
    """入力データ。説明変数たち。"""

    policy_data = []
    """moveのデータ。目的変数（ターゲットデータ）たち。"""

    value_data = []
    """勝敗のデータ。目的変数？たち。？これが 0 のとき学習しない？"""

    kifu_counter = 1
    """npzファイルに書き込む棋譜データの個数を数えておく。npzにも書き込む。"""

    data_counter = 0
    """f"data/sl_data_{data_counter}"""

    cnt = 0###############
    """デバグ用。これまでの棋譜の総数"""

    # 局のループ
    for kifu_path in kifu_list:
        cnt += 1###############
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""

        # 手のループ
        for pos in sgf.get_moves():
            # 対称形でかさ増し
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)

            # 手を一手進める
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            # input_data の局面の手番が勝者の場合は 2 にする。
            value_label = 2 - value_label

        # データセットのサイズを超えたら保存
        if len(value_data) >= DATA_SET_SIZE:
            # データを保存
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)

            # 保存したデータを削除
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]

            kifu_counter = 1
            data_counter += 1




        kifu_counter += 1


    # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)

# pylint: disable=R0914
def generate_supervised_learning_data2(program_dir: str, kifu_dir: str, num_worker: int, board_size: int) -> None:
    """教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。{kifu_dir}/*.sgf。* は 1 始まり。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
    """
    assert kifu_dir is not None, "kifu_dir is None."
    assert program_dir is not None, "program_dir is None."


    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_data start")####################
    print(f"    BATCH_SIZE: {BATCH_SIZE}")
    print(f"    DATA_SET_SIZE: {DATA_SET_SIZE}")
    kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
    print(f"    kifu_num: {kifu_num}")#############


    dt_watch = datetime.datetime.now()################

    board = GoBoard(board_size=board_size)

    input_data = []
    """入力データ。説明変数たち。"""

    policy_data = []
    """moveのデータ。目的変数（ターゲットデータ）たち。"""

    value_data = []
    """勝敗のデータ。目的変数？たち。？これが 0 のとき学習しない？"""

    kifu_counter = 1
    """npzファイルに書き込む棋譜データの個数を数えておく。npzにも書き込む。"""

    data_counter = 0
    """f"data/sl_data_{data_counter}"""

    cnt = 0###############
    """デバグ用。これまでの棋譜の総数"""



    list_kifu_main = glob.glob(os.path.join(kifu_dir, "*.sgf"))

    list_kifu_shuffled = random.sample(list_kifu_main, len(list_kifu_main))

    # それぞれのワーカーに渡すパスのリストのリストの作成。
    list_list_kifu_shuffled = [[]] * num_worker

    # list_kifu_shuffled の先頭から順に num_div ずつに分割すればいい感じになる。
    # num_div % DATA_SET_SIZE == 0 && num_worker - 1 <= len(list_kifu_shuffled) / num_div <= num_worker
    num_div = -((-len(list_kifu_shuffled) // DATA_SET_SIZE) // num_worker) * DATA_SET_SIZE

    for i in range(len(list_list_kifu_shuffled)):
        list_list_kifu_shuffled[i] = list_kifu_shuffled[num_div * i : num_div * (i + 1)]


    # 局のループ
    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        cnt += 1###############
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""

        # 手のループ
        for pos in sgf.get_moves():
            # 対称形でかさ増し
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)

            # 手を一手進める
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            # input_data の局面の手番が勝者の場合は 2 にする。
            value_label = 2 - value_label

        # データセットのサイズを超えたら保存
        if len(value_data) >= DATA_SET_SIZE:
            # データを保存
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)

            # 保存したデータを削除
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]

            kifu_counter = 1
            data_counter += 1


            print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_npz")#####################
            print(f"    saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})")
            print(f"    cnt: {cnt} / {kifu_num}kyoku")
            dt_watch = datetime.datetime.now()


        kifu_counter += 1


    # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)






