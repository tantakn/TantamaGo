"""学習データの生成処理。

value は現手番の勝敗を表す。現手番が勝者の場合は 2、持碁の場合は 1、負けの場合は 0 とする。
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

import sys
import time, datetime################


# import cProfile
# profiler = cProfile.Profile()
# profiler.enable()
import traceback
import logging
import signal
logging.basicConfig(level=logging.INFO)
# タイムアウト時に発生させる例外
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("処理がタイムアウトしました。")


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
def generate_supervised_learning_data(program_dir: str=None, kifu_dir: str=None, board_size: int=9, opt: str="") -> None:

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

    # dbg = 0####################
    # 局のループ
    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        # dbg += 1####################
        # if not dbg % 100:####################
        #     print(sys.getsizeof(value_data) / 1024 /1024, len(value_data), kifu_path)###################
        
        bk_input_data = input_data.copy()
        bk_policy_data = policy_data.copy()
        bk_value_data = value_data.copy()
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""

        # タイムアウト時間を設定（秒）
        timeout_seconds = 1
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            # 手のループ
            for pos in sgf.get_moves():
                # 対称形でかさ増し
                for sym in range(8):
                    input_data.append(generate_input_planes(board, color, sym, opt))
                    policy_data.append(generate_target_data(board, pos, sym))
                    value_data.append(value_label)

                # 手を一手進める
                board.put_stone(pos, color)
                color = Stone.get_opponent_color(color)
                # Valueのラベルを入れ替える。
                # input_data の局面の手番が勝者の場合は 2 にする。
                value_label = 2 - value_label

        except KeyboardInterrupt:
            print("処理が中断されました。現在のスタックトレース:")
            traceback.print_exc()
            logging.info("データ生成が中断されました。")
            raise
        except TimeoutException as e:
            print(e)
            logging.error("処理がタイムアウトしました。")
            logging.error(kifu_path)
            # データを元に戻す
            input_data = bk_input_data.copy()
            policy_data = bk_policy_data.copy()
            value_data = bk_value_data.copy()
            continue
        finally:
            # アラームをリセット
            signal.alarm(0)
        

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
            # dbg = 0####################


            print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_npz")#####################
            print(f"    saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})")
            print(f"    cnt: {cnt} / {kifu_num}kyoku")
            dt_watch = datetime.datetime.now()


        kifu_counter += 1
        cnt += 1###############


    # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)




# # pylint: disable=R0914
# def generate_supervised_learning_data(program_dir: str=None, kifu_dir: str=None, board_size: int=9, opt: str="") -> None:
#     """教師あり学習のデータを生成して保存する。

#     Args:
#         program_dir (str): プログラムのホームディレクトリのパス。
#         kifu_dir (str): SGFファイルを格納しているディレクトリのパス。{kifu_dir}/*.sgf。* は 1 始まり。
#         board_size (int, optional): 碁盤のサイズ. Defaults to 9.
#     """
#     assert kifu_dir is not None, "kifu_dir is None."
#     assert program_dir is not None, "program_dir is None."


#     print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_data start")####################
#     print(f"    BATCH_SIZE: {BATCH_SIZE}")
#     print(f"    DATA_SET_SIZE: {DATA_SET_SIZE}")
#     kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
#     print(f"    kifu_num: {kifu_num}")#############


#     dt_watch = datetime.datetime.now()################

#     board = GoBoard(board_size=board_size)

#     input_data = []
#     """入力データ。説明変数たち。"""

#     policy_data = []
#     """moveのデータ。目的変数（ターゲットデータ）たち。"""

#     value_data = []
#     """勝敗のデータ。目的変数？たち。"""

#     kifu_counter = 1
#     """npzファイルに書き込む棋譜データの個数を数えておく。npzにも書き込む。"""

#     data_counter = 0
#     """f"data/sl_data_{data_counter}"""

#     cnt = 0###############
#     """デバグ用。これまでの棋譜の総数"""

#     dbg = 0####################
#     # 局のループ
#     for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
#         dbg += 1####################
#         if not dbg % 100:####################
#             print(sys.getsizeof(value_data) / 1024 /1024, len(value_data), kifu_path)###################
#         cnt += 1###############
#         board.clear()
#         # ここで勝敗とかも取得してる
#         sgf = SGFReader(kifu_path, board_size)
#         color = Stone.BLACK
#         value_label = sgf.get_value_label()
#         """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""

#         # 手のループ
#         for pos in sgf.get_moves():
#             # 対称形でかさ増し
#             for sym in range(8):
#                 input_data.append(generate_input_planes(board, color, sym, opt))
#                 policy_data.append(generate_target_data(board, pos, sym))
#                 value_data.append(value_label)

#             # 手を一手進める
#             board.put_stone(pos, color)
#             color = Stone.get_opponent_color(color)
#             # Valueのラベルを入れ替える。
#             # input_data の局面の手番が勝者の場合は 2 にする。
#             value_label = 2 - value_label

#         # データセットのサイズを超えたら保存
#         if len(value_data) >= DATA_SET_SIZE:
#             # データを保存
#             _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)

#             # 保存したデータを削除
#             input_data = input_data[DATA_SET_SIZE:]
#             policy_data = policy_data[DATA_SET_SIZE:]
#             value_data = value_data[DATA_SET_SIZE:]

#             kifu_counter = 1
#             data_counter += 1


#             print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_npz")#####################
#             print(f"    saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})")
#             print(f"    cnt: {cnt} / {kifu_num}kyoku")
#             dt_watch = datetime.datetime.now()


#         kifu_counter += 1


#     # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
#     n_batches = len(value_data) // BATCH_SIZE
#     if n_batches > 0:
#         _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)



def generate_reinforcement_learning_data(program_dir: str, kifu_dir_list: List[str], board_size: int=9, input_opt: str="") -> None:
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
                input_data.append(generate_input_planes(board, color, sym, input_opt))
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





import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

def generate_supervised_learning_data_mt(program_dir: str=None, kifu_dir: str=None, board_size: int=9, opt: str="", n_threads: int=10) -> None:
    """マルチスレッドで教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。{kifu_dir}/*.sgf。* は 1 始まり。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
        opt (str, optional): 追加オプション. Defaults to "".
        n_threads (int, optional): 使用するスレッド数. Defaults to 4.
    """
    assert kifu_dir is not None, "kifu_dir is None."
    assert program_dir is not None, "program_dir is None."

    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_data_mt start (threads: {n_threads})")
    print(f"    BATCH_SIZE: {BATCH_SIZE}")
    print(f"    DATA_SET_SIZE: {DATA_SET_SIZE}")

    # 棋譜ファイルのリストを取得
    kifu_files = sorted(glob.glob(os.path.join(kifu_dir, "*.sgf")))
    kifu_num = len(kifu_files)
    print(f"    kifu_num: {kifu_num}")
    
    # スレッド間で共有するカウンターとロック
    data_counter_lock = threading.Lock()
    data_counter = [0]  # リストで包むことでミュータブルにする
    processed_files_counter = [0]
    
    # 処理済みの棋譜をカウントする関数
    def increment_processed_counter():
        with data_counter_lock:
            processed_files_counter[0] += 1
            return processed_files_counter[0]
    
    # データを保存する関数（スレッドセーフ）
    def save_data_thread_safe(input_data, policy_data, value_data, kifu_counter):
        with data_counter_lock:
            save_file_path = os.path.join(program_dir, "data", f"sl_data_{data_counter[0]}")
            _save_data(save_file_path, input_data, policy_data, value_data, kifu_counter)
            data_counter[0] += 1
            return data_counter[0] - 1
    
    # 各スレッドで実行される関数
    def process_kifu_batch(kifu_batch):
        board = GoBoard(board_size=board_size)
        input_data = []
        policy_data = []
        value_data = []
        kifu_counter = 0
        dt_watch = datetime.datetime.now()

        for kifu_path in kifu_batch:
            kifu_counter += 1
            bk_input_data = input_data.copy()
            bk_policy_data = policy_data.copy()
            bk_value_data = value_data.copy()
            
            board.clear()
            
            # タイムアウト処理用のフラグと変数
            timeout_occurred = [False]
            processing_done = [False]
            
            def handle_timeout():
                if not processing_done[0]:
                    timeout_occurred[0] = True
                    logging.error(f"処理がタイムアウトしました: {kifu_path}")
            
            # タイマーを設定（秒）
            timeout_seconds = 1
            timer = threading.Timer(timeout_seconds, handle_timeout)
            timer.start()
            
            try:
                # ここで勝敗とかも取得してる
                sgf = SGFReader(kifu_path, board_size)
                color = Stone.BLACK
                value_label = sgf.get_value_label()
                
                # 手のループ
                for pos in sgf.get_moves():
                    # タイムアウトチェック
                    if timeout_occurred[0]:
                        # データを元に戻す
                        input_data = bk_input_data.copy()
                        policy_data = bk_policy_data.copy()
                        value_data = bk_value_data.copy()
                        break
                    
                    # 対称形でかさ増し
                    for sym in range(8):
                        input_data.append(generate_input_planes(board, color, sym, opt))
                        policy_data.append(generate_target_data(board, pos, sym))
                        value_data.append(value_label)

                    # 手を一手進める
                    board.put_stone(pos, color)
                    color = Stone.get_opponent_color(color)
                    # Valueのラベルを入れ替える
                    value_label = 2 - value_label
                    
            except Exception as e:
                logging.error(f"処理中にエラーが発生しました: {kifu_path}, {str(e)}")
                # データを元に戻す
                input_data = bk_input_data.copy()
                policy_data = bk_policy_data.copy()
                value_data = bk_value_data.copy()
                
            finally:
                # 処理完了フラグを立てる
                processing_done[0] = True
                # タイマーをキャンセル
                timer.cancel()
            
            # タイムアウトが発生した場合は次のファイルへ
            if timeout_occurred[0]:
                continue
            
            # 処理済みファイル数を更新
            processed = increment_processed_counter()
            if processed % 100 == 0:
                print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] 処理済み: {processed}/{kifu_num} 棋譜")

            # データセットのサイズを超えたら保存
            if len(value_data) >= DATA_SET_SIZE:
                saved_counter = save_data_thread_safe(input_data, policy_data, value_data, kifu_counter)
                print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_npz")
                print(f"    saved: sl_data_{saved_counter}.npz ({datetime.datetime.now() - dt_watch})")
                print(f"    処理済み: {processed}/{kifu_num}棋譜")
                
                # 保存したデータを削除
                input_data = input_data[DATA_SET_SIZE:]
                policy_data = policy_data[DATA_SET_SIZE:]
                value_data = value_data[DATA_SET_SIZE:]
                kifu_counter = 0
                dt_watch = datetime.datetime.now()

        # 端数の出力。BATCH_SIZE で割り切れる数だけデータを保存する。他は捨てる。
        n_batches = len(value_data) // BATCH_SIZE
        if n_batches > 0:
            save_data_thread_safe(input_data[0:n_batches*BATCH_SIZE], 
                                policy_data[0:n_batches*BATCH_SIZE], 
                                value_data[0:n_batches*BATCH_SIZE], 
                                kifu_counter)

    # 棋譜ファイルをスレッド数に応じて分割
    kifu_batches = []
    batch_size = (len(kifu_files) + n_threads - 1) // n_threads
    
    for i in range(0, len(kifu_files), batch_size):
        kifu_batches.append(kifu_files[i:i + batch_size])
    
    # ThreadPoolExecutorを使って並列処理
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = []
        for batch in kifu_batches:
            futures.append(executor.submit(process_kifu_batch, batch))
        
        # 全てのスレッドが完了するのを待つ
        for future in futures:
            future.result()
    
    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] gen_sl_data_mt finished")
    print(f"    全ての棋譜処理完了: {processed_files_counter[0]}/{kifu_num}")
    print(f"    生成されたデータセット数: {data_counter[0]}")



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



"""
tmp_npz = tmp_load_data_set("/home0/y2024/u2424004/igo/TantamaGo/backup/data_Q50000/sl_data_0.npz")

推論するときは
input_t = tmp_npz[0][100].unsqueeze(0).to(device)
みたいにバッチ次元を追加してデバイスに送る。

input_t = tmp_npz[0]
print(input_t.shape)
# torch.Size([1024000, 6, 9, 9])]

input_t = tmp_npz[0][100]
print(input_t.shape)######
print(input_t)######
torch.Size([6, 9, 9])
#=>
tensor([[[1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 0., 1., 0., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 0., 1., 1., 1., 1.],
         [1., 1., 0., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 0., 1., 1., 0., 1., 1.],
         [1., 1., 0., 0., 0., 0., 0., 1., 1.],
         [1., 1., 0., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 1., 1., 1., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0., 1., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0., 0., 0., 0.]],

        [[1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.],
         [1., 1., 1., 1., 1., 1., 1., 1., 1.]]])
torch.Size([6, 9, 9])
"""