import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import logging
mylog = logging.getLogger("mylog")
mylog.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)





# # nn：調整すべきパラメータを持つ関数たち
# import torch.nn as nn
# # F：調整すべきパラメータを持たない関数たち
# import torch.nn.functional as F

# # 乱数シードの固定
# torch.manual_seed(1)

# # 普通は float32 で作る。ドット打つと指定できる。
# x = torch.tensor([[1., 2., 3.]])
# print(x.dtype)
# # torch.float32

# # 線形変換たち。初期値はランダム。
# fc1 = nn.Linear(3, 2)
# fc2 = nn.Linear(2, 1)

# # 重みとバイアスの確認
# print(fc1.weight)
# print(fc1.bias)
# # tensor([[ 0.2975, -0.2548, -0.1119],
# #         [ 0.2710, -0.5435,  0.3462]], requires_grad=True)
# # Parameter containing:
# # tensor([-0.1188,  0.2937], requires_grad=True)

# # 線形変換の実行
# u1 = fc1(x)
# z1 = F.relu(u1)
# y = fc2(z1)

# print(y)
# # tensor([[0.1514]], grad_fn=<AddmmBackward0>)

# # 目標値
# t = torch.tensor([[1.]])

# # 平均二乗誤差
# loss = F.mse_loss(t, y)

# print(loss)
# # tensor(0.7201, grad_fn=<MseLossBackward0>)


# from sklearn.datasets import load_iris


# import torch
# import numpy as np

# from nn.network.dual_net import DualNet

# model_file_path = "model/sl-model_default.bin"
# model_file_path2 = "model/sl-model_20240912_011228_Ep:14.bin"

# device = torch.device("cpu")
# network = DualNet(device)
# network.to(torch.device("cpu"))
# try:
#     network.load_state_dict(torch.load(model_file_path))
# except Exception as e: # pylint: disable=W0702
#     print(f"👺Failed to load {model_file_path}. : ", e)
#     raise e

# # たぶん、レイヤーの行列の名前のリストが返ってくる
# # print(network.state_dict().keys())

# # # たぶん、レイヤーの行列の値が全部返ってくる
# # print(network.state_dict())

# # # たぶん、レイヤーの行列の値が全部返ってくる
# # print(list(network.parameters()))

# print(network)



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


import time, datetime#################


def _save_data(save_file_path: str, input_data: np.ndarray, policy_data: np.ndarray, value_data: np.ndarray, kifu_counter: int) -> None:
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
def generate_supervised_learning_data(program_dir: str, kifu_dir: str, board_size: int=9) -> None:
    """教師あり学習のデータを生成して保存する。

    Args:
        program_dir (str): プログラムのホームディレクトリのパス。
        kifu_dir (str): SGFファイルを格納しているディレクトリのパス。
        board_size (int, optional): 碁盤のサイズ. Defaults to 9.
    """
    dt_watch = datetime.datetime.now()################
    print(f"🐾generate_supervised_learning_data {dt_watch}🐾")############

    board = GoBoard(board_size=board_size)

    input_data = []
    policy_data = []
    value_data = []

    kifu_counter = 1
    data_counter = 0

    kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
    print(f"kifu_num: {kifu_num}")#############

    # 局のループ
    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け"""

        # 手のループ
        for pos in sgf.get_moves():
            # 対称形でかさ増し
            for sym in range(8):
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            value_label = 2 - value_label

        if len(value_data) >= DATA_SET_SIZE:
            _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)
            input_data = input_data[DATA_SET_SIZE:]
            policy_data = policy_data[DATA_SET_SIZE:]
            value_data = value_data[DATA_SET_SIZE:]
            kifu_counter = 1
            data_counter += 1

            print(f"""
saved: sl_data_{data_counter}.npz ({datetime.datetime.now() - dt_watch})
from: {kifu_path} / {kifu_num}kyoku""")#####################
            dt_watch = datetime.datetime.now()

        kifu_counter += 1

    print("qwer")##########

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "data", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)


npz = np.load("data/sl_data_0.npz")

print(npz["input"].shape)
print(npz["policy"].shape)
print(npz["value"].shape)