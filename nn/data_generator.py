"""学習データの生成処理。
"""
import glob
import os
import random
from typing import List, NoReturn
import numpy as np
from board.go_board import GoBoard
from board.stone import Stone
from nn.feature import generate_input_planes, generate_target_data, \
    generate_rl_target_data
from sgf.reader import SGFReader
from learning_param import BATCH_SIZE, DATA_SET_SIZE

import time, datetime#################


def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray,\
    value_data: np.ndarray, kifu_counter: int) -> NoReturn:
    """学習データをnpzファイルとして出力する。

    Args:
        save_file_path (str): 保存するファイルパス。
        input_data (np.ndarray): 入力データ。
        policy_data (np.ndarray): Policyのデータ。
        value_data (np.ndarray): Valueのデータ
        kifu_counter (int): データセットにある棋譜データの個数。
    """
    save_data = {
        "input": np.array(input_data[0:DATA_SET_SIZE]),
        "policy": np.array(policy_data[0:DATA_SET_SIZE]),
        "value": np.array(value_data[0:DATA_SET_SIZE], dtype=np.int32),
        "kifu_count": np.array(kifu_counter)
    }
    np.savez_compressed(save_file_path, **save_data)

# pylint: disable=R0914
def generate_supervised_learning_data(program_dir: str, kifu_dir: str, \
    board_size: int=9) -> NoReturn:
    """教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
    """
    dt_watch = datetime.datetime.now()
    print(f"🐾generate_supervised_learning_data {dt_watch}🐾")############

    board = GoBoard(board_size=board_size)

    input_data = []
    policy_data = []
    value_data = []

    kifu_counter = 1
    data_counter = 0

    kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
    print(f"kifu_num: {kifu_num}")#############

    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        board.clear()
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()

        for pos in sgf.get_moves():
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            value_label = 2 - value_label

        if len(value_data) >= DATA_SET_SIZE:
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
                input_data, policy_data, value_data, kifu_counter)
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]
            kifu_counter = 1
            data_counter += 1

            print(f"""\
    saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})
    from: {kifu_path} / {kifu_num}\
""")#####################
            dt_watch = datetime.datetime.now()

        kifu_counter += 1
    print("qwer")

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], kifu_counter)


def generate_reinforcement_learning_data(program_dir: str, kifu_dir_list: List[str], \
    board_size: int=9) -> NoReturn:
    """強化学習で使用するデータを生成し、保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリ。
        kifu_dir_list (List[str]): 棋譜ファイルを保存しているディレクトリパスのリスト。
        board_size (int, optional): 碁盤の大きさ。デフォルトは9。
    """
    dt_watch = datetime.datetime.now()
    print(f"🐾generate_reinforcement_learning_data {dt_watch}🐾")##################

    board = GoBoard(board_size=board_size)

    input_data = []
    policy_data = []
    value_data = []

    kifu_counter = 1
    data_counter = 0

    kifu_list = []
    for kifu_dir in kifu_dir_list:
        # kifu_list.extend(glob.glob(os.path.join(kifu_dir, "*", "*.sgf"))) # natukazeの棋譜のフォルダ形式に合うようにした
        kifu_list.extend(glob.glob(os.path.join(kifu_dir, "*.sgf")))
    random.shuffle(kifu_list)

    for kifu_path in kifu_list:
        board.clear()
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        target_index = sorted(np.random.permutation(np.arange(sgf.get_n_moves()))[:8]) # 総手数以下の数からランダムで８個選んで選んだのをソート
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
            _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), \
                input_data, policy_data, value_data, kifu_counter)
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]
            kifu_counter = 1
            data_counter += 1

        kifu_counter += 1

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"rl_data_{data_counter}"), \
            input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], \
            value_data[0:n_batches*BATCH_SIZE], kifu_counter)
