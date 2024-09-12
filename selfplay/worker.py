"""自己対戦実行ワーカの実装。
"""
import glob
import os
import random
import time
from typing import List
import numpy as np

from board.constant import PASS, RESIGN
from board.go_board import GoBoard, copy_board
from board.stone import Stone

from sgf.selfplay_record import SelfPlayRecord
from mcts.tree import MCTSTree
from mcts.time_manager import TimeManager, TimeControl
from nn.utility import load_network, load_DualNet_128_12
from learning_param import SELF_PLAY_VISITS

import psutil, subprocess, datetime

def choose_network(network_name: str, model_file_path: str, use_gpu: bool):
    if network_name == "DualNet":
        network = load_network(model_file_path=model_file_path, use_gpu=use_gpu)
    elif network_name == "DualNet_128_12":
        network = load_DualNet_128_12(model_file_path=model_file_path, use_gpu=use_gpu)
    else:
        print(f"👺network_name: {network_name} is not defined.")
        raise(f"network_name is not defined.")
    return network


# pylint: disable=R0913,R0914
def selfplay_worker(save_dir: str, model_file_path: str, index_list: List[int], size: int, visits: int, use_gpu: bool, network_name1: str) -> None:
    """自己対戦実行ワーカ。

    Args:
        save_dir (str): 棋譜ファイルを保存するディレクトリパス。
        model_file_path (str): 使用するニューラルネットワークモデルファイルパス。
        index_list (List[int]): 棋譜ファイル保存時に使用するインデックスリスト。
        size (int): 碁盤の大きさ。
        visits (int): 自己対戦実行時の探索回数。
        use_gpu (bool): GPU使用フラグ。
        network_name1 (str): 使用するニューラルネットワーク名。
    """

    # print("🐾selfplay_worker_start")##############

    board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    init_board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    """初期化用"""
    record = SelfPlayRecord(save_dir, board.coordinate)

    network = choose_network(network_name1, model_file_path, use_gpu)

    network.training = False

    np.random.seed(random.choice(index_list))

    mcts = MCTSTree(network, tree_size=SELF_PLAY_VISITS * 10)
    time_manager = TimeManager(TimeControl.CONSTANT_PLAYOUT, constant_visits=visits)

    max_moves = (board.get_board_size() ** 2) * 2

    for index in index_list:
        if os.path.isfile(os.path.join(save_dir, f"{index}.sgf")):
            continue
        copy_board(board, init_board)
        color = Stone.BLACK
        record.clear()
        pass_count = 0
        never_resign = True if random.randint(1, 10) == 1 else False # pylint: disable=R1719
        is_resign = False
        score = 0.0
        for _ in range(max_moves):
            pos = mcts.generate_move_with_sequential_halving(board=board, color=color, time_manager=time_manager, never_resign=never_resign)

            if pos == RESIGN:
                winner = Stone.get_opponent_color(color)
                is_resign = True
                break

            board.put_stone(pos, color)

            if pos == PASS:
                pass_count += 1
            else:
                pass_count = 0

            record.save_record(mcts.get_root(), pos, color)

            color = Stone.get_opponent_color(color)

            if pass_count == 2:
                winner = Stone.EMPTY
                break

        if pass_count == 2:
            score = board.count_score() - board.get_komi()
            if score > 0.1:
                winner = Stone.BLACK
            elif score < -0.1:
                winner = Stone.WHITE
            else:
                winner = Stone.OUT_OF_BOARD

        record.set_index(index)
        record.write_record(winner, board.get_komi(), is_resign, score)


def display_selfplay_progress_worker(save_dir: str, num_data: int, use_gpu: bool) -> None:
    """自己対戦の進捗を表示する。

    Args:
        save_dir (str): 生成した棋譜ファイルが保存されるディレクトリのパス。
    """
    start_time = time.time()
    while True:
        time.sleep(60)
        current_num_data = len(glob.glob(os.path.join(save_dir, "*.sgf")))
        current_time = time.time()

        msg = f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] generating\n{current_num_data:5d}/{num_data:5d} games "
        msg += f"({3600 * current_num_data / (current_time - start_time):.4f} games/hour)."
        print(msg)




# pylint: disable=R0913,R0914
def selfplay_worker_vs(save_dir: str, model_file_path1: str, model_file_path2: str, index_list: List[int], size: int, visits: int, use_gpu: bool, network_name1: str, network_name2: str) -> None:
    """異なるモデルを対戦させる自己対戦実行ワーカ。

    Args:
        save_dir (str): 棋譜ファイルを保存するディレクトリパス。
        model_file_path1 (str): 使用するニューラルネットワークモデルファイルパス。
        model_file_path2 (str): 使用するニューラルネットワークモデルファイルパス2。
        index_list (List[int]): 棋譜ファイル保存時に使用するインデックスリスト。
        size (int): 碁盤の大きさ。
        visits (int): 自己対戦実行時の探索回数。
        use_gpu (bool): GPU使用フラグ。
        network_name1 (str): 使用するニューラルネットワーク名。
        network_name2 (str): 使用するニューラルネットワーク名2。
    """
    # print("🐾selfplay_worker_vs_start")##############

    board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    init_board = GoBoard(board_size=size, komi=7.0, check_superko=True)
    record = SelfPlayRecord(save_dir, board.coordinate)

    network1 = choose_network(network_name1, model_file_path1, use_gpu)
    network2 = choose_network(network_name2, model_file_path2, use_gpu)

    network1.training = False
    network2.training = False

    np.random.seed(random.choice(index_list))

    mcts1 = MCTSTree(network1, tree_size=SELF_PLAY_VISITS * 10)
    mcts2 = MCTSTree(network2, tree_size=SELF_PLAY_VISITS * 10)
    time_manager = TimeManager(TimeControl.CONSTANT_PLAYOUT, constant_visits=visits)

    max_moves = (board.get_board_size() ** 2) * 2

    for index in index_list:
        if os.path.isfile(os.path.join(save_dir, f"{index}.sgf")):
            continue
        copy_board(board, init_board)
        color = Stone.BLACK
        record.clear()
        pass_count = 0
        never_resign = True if random.randint(1, 10) == 1 else False # pylint: disable=R1719
        is_resign = False
        score = 0.0
        for i in range(max_moves):
            # index が奇数のときは model_file_path1 が先手、偶数のときは model_file_path2 が先手
            if (i + index) % 2 == 1:
                mcts = mcts1
            else:
                mcts = mcts2

            pos = mcts.generate_move_with_sequential_halving(board=board, color=color, time_manager=time_manager, never_resign=never_resign)

            if pos == RESIGN:
                winner = Stone.get_opponent_color(color)
                is_resign = True
                break

            board.put_stone(pos, color)

            if pos == PASS:
                pass_count += 1
            else:
                pass_count = 0

            record.save_record(mcts.get_root(), pos, color)

            color = Stone.get_opponent_color(color)

            if pass_count == 2:
                winner = Stone.EMPTY
                break

        if pass_count == 2:
            score = board.count_score() - board.get_komi()
            if score > 0.1:
                winner = Stone.BLACK
            elif score < -0.1:
                winner = Stone.WHITE
            else:
                winner = Stone.OUT_OF_BOARD

        record.set_index(index)
        # record.write_record(winner, board.get_komi(), is_resign, score)

        tmp_path1 = model_file_path1
        tmp_path2 = model_file_path2
        if model_file_path1 == model_file_path2:
            tmp_path1 += "_selfVs1"
            tmp_path2 += "_selfVs2"

        if (index) % 2 == 1:
            record.write_record(winner, board.get_komi(), is_resign, score, black_name=tmp_path1, white_name=tmp_path2)

        else:
            record.write_record(winner, board.get_komi(), is_resign, score, black_name=tmp_path2, white_name=tmp_path1)


