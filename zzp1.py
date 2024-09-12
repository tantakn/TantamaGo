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

# narr = np.random.randint(0, 10, (2, 3))
# print(narr)
# # [[9 7 5]
# #  [7 9 7]]

# narr = np.random.randint(0, 10, (2, 3, 4, 5))
# print(narr)
# # [[[0 2 8 5]
# #   [7 7 6 1]
# #   [3 4 1 0]]

# #  [[2 5 7 9]
# #   [5 9 7 4]
# #   [4 8 8 1]]]

# print(np.sum(narr))

# score = np.random.dirichlet(alpha=np.ones(10))
# print(score)

# a = np.arange(24)
# print(a)
# # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]

# # 4 * 6 = 24 だから変換できる。
# print(a.reshape(4, 6))
# # [[ 0  1  2  3  4  5]
# #  [ 6  7  8  9 10 11]
# #  [12 13 14 15 16 17]
# #  [18 19 20 21 22 23]]

# # 2 * 3 * 4 = 24 だから変換できる。
# print(a.reshape(2, 3, 4))
# # [[[ 0  1  2  3]
# #   [ 4  5  6  7]
# #   [ 8  9 10 11]]

# #  [[12 13 14 15]
# #   [16 17 18 19]
# #   [20 21 22 23]]]

# # 変換の順番（？）を指定できる。
# print(a.reshape(4, 6, order='F'))
# # [[ 0  4  8 12 16 20]
# #  [ 1  5  9 13 17 21]
# #  [ 2  6 10 14 18 22]
# #  [ 3  7 11 15 19 23]]

# # -1 を指定すると、自動で計算してくれる。
# print(a.reshape(-1, 6))
# # [[ 0  1  2  3  4  5]
# #  [ 6  7  8  9 10 11]
# #  [12 13 14 15 16 17]
# #  [18 19 20 21 22 23]]

# b = a.copy().reshape(2, 3, 4)
# """setumri"""

# b[1][1][1] = 100
# print(b)
# print(a)

# import sympy as sp


# # spで宣言
# x = sp.symbols('x')

# # 微分
# print(sp.diff(x ** 2 * 3 + x, x))
# # 6*x + 1

# # 関数で式を宣言しても大丈夫
# def f (x):
#     return x ** 2 + 4 * x + 7
# print(sp.diff(f(x), x))
# # 2*x + 4

# # 偏微分
# y = sp.symbols('y')
# print(sp.diff(x ** 3 * y + y, x))
# # 3*x**2*y

# # 積分
# print(sp.integrate(x ** 2 * 3 + x, x))
# # 6*x + 1

# # sin とか有名定数はインポートすれば使える
# from sympy import sin, pi

# # 定積分
# print(sp.integrate(sin(x), (x, 0, pi/2)))
# # 1

# # 式として積分できなくても、evalf()で数値積分できる
# print(sp.integrate(sin(sin(x)), (x, 0, pi)).evalf())
# # 1.78648748195005

# # 方程式 x**2 - x - 6 = 0 を解く
# print(sp.solve(x**2 - x - 6, x))
# # [-2, 3]

# # 連立方程式も解ける
# print(sp.solve([x + y - 1, x - y - 1], [x, y]))
# # {x: 1, y: 0}

# # 1 付近から 8 桁の精度で、解を探す
# print(sp.nsolve(x**2 - x - 6, x, 1, prec=8))




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
    print("!!!!!!!!save")########
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
    """入力データ。説明変数たち。"""
    policy_data = []
    """moveのデータ。目的変数たち。"""
    value_data = []
    """勝敗のデータ。目的変数？たち。？これが 0 のとき学習しない？"""

    kifu_counter = 1
    data_counter = 0
    """f"data/sl_data_{data_counter}"""

    kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
    print(f"kifu_num: {kifu_num}")#############

    ccc = 0#################
    # 局のループ
    for kifu_path in sorted(glob.glob(os.path.join(kifu_dir, "*.sgf"))):
        board.clear()
        # ここで勝敗とかも取得してる
        sgf = SGFReader(kifu_path, board_size)
        color = Stone.BLACK
        value_label = sgf.get_value_label()
        """勝ち負け。黒勝ちは2、白勝ちは0、持碁は1。"""
        cnt = 0#################
        # 手のループ
        for pos in sgf.get_moves():
            cnt += 1##################
            # 対称形でかさ増し
            for sym in range(8):
                ccc += 1##################
                if sym == 0 and cnt < 10:###############
                    print(value_label)
                    print("-----------------")
                input_data.append(generate_input_planes(board, color, sym))
                policy_data.append(generate_target_data(board, pos, sym))
                value_data.append(value_label)
            board.put_stone(pos, color)
            color = Stone.get_opponent_color(color)
            # Valueのラベルを入れ替える。
            # input_data の局面の手番が勝者の場合は 2 にする。
            value_label = 2 - value_label
    
    
        print(f"kifu_counter: {kifu_counter}")#################
        print(f"data_counter: {data_counter}")#################
        print(f"cnt: {cnt}")#################

        if len(value_data) >= DATA_SET_SIZE:
            _save_data(os.path.join(program_dir, "backup", "test", f"sl_data_{data_counter}"), input_data, policy_data, value_data, kifu_counter)
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
    print(f"ccc: {ccc}")#################

    # 端数の出力
    n_batches = len(value_data) // BATCH_SIZE
    if n_batches > 0:
        _save_data(os.path.join(program_dir, "backup", "test", f"sl_data_{data_counter}"), input_data[0:n_batches*BATCH_SIZE], policy_data[0:n_batches*BATCH_SIZE], value_data[0:n_batches*BATCH_SIZE], kifu_counter)

generate_supervised_learning_data(os.path.dirname(__file__), "archive/2")#################


# npz = np.load("backup/test/sl_data_0.npz")

# print(npz["input"].shape)
# print(npz["policy"].shape)
# print(npz["value"].shape)

