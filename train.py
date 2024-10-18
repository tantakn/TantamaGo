"""教師あり学習のエントリーポイント。
"""
import glob
import os
import click
from learning_param import BATCH_SIZE, EPOCHS
from board.constant import BOARD_SIZE
from nn.learn import train_on_cpu, train_on_gpu, train_with_gumbel_alphazero_on_gpu, train_with_gumbel_alphazero_on_cpu,  train_on_gpu_ddp
from nn.data_generator import generate_supervised_learning_data, generate_reinforcement_learning_data

import threading, time, datetime
from monitoring import display_train_monitoring_worker

import torch

from nn.utility import split_train_test_set



@click.command()
@click.option('--kifu-dir', type=click.STRING, \
    help="学習データの棋譜ファイルを格納したディレクトリのパス。指定がない場合はデータ生成を実行しない。")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤の大きさ。最小2, 最大{BOARD_SIZE}")
@click.option('--use-gpu', type=click.BOOL, default=True, \
    help="学習時にGPUを使用するフラグ。指定がなければGPUを使用するものとする。")
@click.option('--use-ddp', 'ddp', type=click.BOOL, default=False, \
    help="ddp。")#############
@click.option('--rl', type=click.BOOL, default=False, \
    help="強化学習実行フラグ。教師あり学習を実行するときにはfalseを指定する。")
@click.option('--window-size', type=click.INT, default=300000, \
    help="強化学習時のウィンドウサイズ")
@click.option('--net', 'network_name', type=click.STRING, default="DualNet", \
    help="ネットワーク。デフォルトは DualNet。DualNet_256_24 とかを指定する。")
@click.option('--npz-dir', 'npz_dir', type=click.STRING, default="data", \
    help="npzがあるフォルダのパス。デフォルトは data。")
@click.option('--rl-num', 'rl_num', type=click.INT, default=-1, \
    help="rl のパイプラインが何周目か。")
@click.option('--rl-datetime', 'rl_datetime', type=click.STRING, default="", \
    help="rl のパイプラインの開始日時。")
@click.option('--input-opt', 'input_opt', type=click.STRING, default="", \
    help="input_planes のオプション。")
def train_main(kifu_dir: str, size: int, use_gpu: bool, rl: bool, window_size: int, network_name: str, npz_dir: str, ddp: bool, rl_num: int, rl_datetime: str, input_opt: str): # pylint: disable=C0103
    """教師あり学習、または強化学習のデータ生成と学習を実行する。

    Args:
        kifu_dir (str): 学習する棋譜ファイルを格納したディレクトリパス。
        size (int): 碁盤の大きさ。
        use_gpu (bool): GPU使用フラグ。
        rl (bool): 強化学習実行フラグ。
        window_size (int): 強化学習で使用するウィンドウサイズ。
    """

    print(f"🐾train_main")########
    print(f"    EPOCHS: {EPOCHS}")
    print(f"    BATCH_SIZE: {BATCH_SIZE}")
    print(f"    kifu_dir: {kifu_dir}")
    print(f"    size: {size}")
    print(f"    use_gpu: {use_gpu}")
    print(f"    rl: {rl}")
    print(f"    window_size: {window_size}")
    print(f"    network_name: {network_name}")
    print(f"    npz_dir: {npz_dir}")
    print(f"    ddp: {ddp}")
    print(f"    rl_num: {rl_num}")
    print(f"    rl_datetime: {rl_datetime}")
    print(f"    input_opt: {input_opt}")

    # ハードウェア使用率の監視スレッドを起動
    monitoring_worker = threading.Thread(target=display_train_monitoring_worker, args=(use_gpu, True, 300), daemon=True)
    monitoring_worker.start()


    program_dir = os.path.dirname(__file__)


    # 学習データの指定がある場合はデータを生成する
    if kifu_dir is not None:
        if rl:
            # rl の kifu_dir は kifu_dir/数字/*.sgf
            kifu_index_list: list[int] = [int(os.path.split(dir_path)[-1]) for dir_path in glob.glob(os.path.join(kifu_dir, "*"))]
            """archive/数字/の数字部分を取得してリストに格納する。"""
            num_kifu = 0
            kifu_dir_list: list[str] = []
            """棋譜のパスのリスト。"""
            for index in sorted(kifu_index_list, reverse=True):
                kifu_dir_path = os.path.join(kifu_dir, str(index))
                num_kifu += len(glob.glob(os.path.join(kifu_dir_path, "*.sgf")))
                kifu_dir_list.append(kifu_dir_path)
                if num_kifu >= window_size:
                    break

            generate_reinforcement_learning_data(program_dir=program_dir, kifu_dir_list=kifu_dir_list, board_size=size)
        else:
            # こっちの kifu_dir は kifu_dir/*.sgf
            generate_supervised_learning_data(program_dir=program_dir, kifu_dir=kifu_dir, board_size=size, opt=input_opt)
            # generate_supervised_learning_data(program_dir=program_dir, kifu_dir=kifu_dir, board_size=size)


    if rl:
        if use_gpu:
            train_with_gumbel_alphazero_on_gpu(program_dir=program_dir, board_size=size, batch_size=BATCH_SIZE, rl_num=rl_num, rl_datetime=rl_datetime, network_name=network_name)
        else:
            train_with_gumbel_alphazero_on_cpu(program_dir=program_dir, board_size=size, batch_size=BATCH_SIZE)
    else:
        if use_gpu and ddp:
            train_on_gpu_ddp(program_dir=program_dir,board_size=size,  batch_size=BATCH_SIZE, epochs=EPOCHS, network_name=network_name, npz_dir=npz_dir)
        elif use_gpu:
            train_on_gpu(program_dir=program_dir,board_size=size,  batch_size=BATCH_SIZE, epochs=EPOCHS, network_name=network_name, npz_dir=npz_dir)
        else:
            train_on_cpu(program_dir=program_dir,board_size=size, batch_size=BATCH_SIZE, epochs=EPOCHS)



if __name__ == "__main__":
    train_main() # pylint: disable=E1120
