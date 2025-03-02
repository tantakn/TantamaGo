"""教師あり学習のエントリーポイント。

npz を作るとき
python3 train.py --size 13 --kifu-dir /home0/y2024/u2424004/igo/TantamaGo/SgfFile/13x13-record-2


学習するとき
python3 train.py --size 13 --use-ddp true --npz-dir data --net DualNet_256_24

[20250226_170831] learn
epoch 0, data-0 : loss = 4.722960, time = 419.1 [s].
        policy loss : 4.715367
        value loss  : 0.759278
[20250226_171151] monitoring
cpu: 8.9% [5.5, 72.4, 1.1, 0.4, 1.1, 23.8, 0.5, 0.3, 0.7, 0.3, 0.6, 0.6] 
mem: 26.8% 🔥
NVIDIA GeForce RTX 3060, 0, 98 %, 6635 MiB, 154.06 W 🔥

[20250226_171533] learn
epoch 0, data-1 : loss = 3.384733, time = 420.3 [s].
        policy loss : 3.376641
        value loss  : 0.809221


チェックポイントから学習するとき
python3 train.py --size 13 --use-ddp true --npz-dir data --net DualNet_256_24 --checkpoint-dir model/checkpoint_20250227_033544_Ep:00.bin
"""
import glob
import os
import click
from learning_param import BATCH_SIZE, EPOCHS
from board.constant import BOARD_SIZE
from nn.learn import train_on_cpu, train_on_gpu, train_with_gumbel_alphazero_on_gpu, train_with_gumbel_alphazero_on_cpu,  train_on_gpu_ddp
from nn.data_generator import generate_supervised_learning_data, generate_reinforcement_learning_data, generate_supervised_learning_data_mt

import threading, time, datetime
from monitoring import display_train_monitoring_worker

import torch

from nn.utility import split_train_test_set

import resource




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
@click.option('--checkpoint-dir', 'checkpoint_dir', type=click.STRING, default=None, \
    help="checkpointがあるフォルダのパス。デフォルトは None。")
@click.option('--rl-num', 'rl_num', type=click.INT, default=-1, \
    help="rl のパイプラインが何周目か。")
@click.option('--rl-datetime', 'rl_datetime', type=click.STRING, default="", \
    help="rl のパイプラインの開始日時。")
@click.option('--input-opt', 'input_opt', type=click.STRING, default="", \
    help="input_planes のオプション。")
def train_main(kifu_dir: str, size: int, use_gpu: bool, rl: bool, window_size: int, network_name: str, npz_dir: str, checkpoint_dir: str, ddp: bool, rl_num: int, rl_datetime: str, input_opt: str): # pylint: disable=C0103
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
    print(f"    checkpoint_dir: {checkpoint_dir}")
    print(f"    ddp: {ddp}")
    print(f"    rl_num: {rl_num}")
    print(f"    rl_datetime: {rl_datetime}")
    print(f"    input_opt: {input_opt}")

    # ハードウェア使用率の監視スレッドを起動
    monitoring_worker = threading.Thread(target=display_train_monitoring_worker, args=(use_gpu, True, 300, os.getpid()), daemon=True)
    monitoring_worker.start()


    # # メモリ使用量を制限（単位：バイト）
    # soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    # resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024 * 8, hard))  # 1GBに制限


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

            generate_reinforcement_learning_data(program_dir=program_dir, kifu_dir_list=kifu_dir_list, board_size=size, input_opt=input_opt)
        else:
            # こっちの kifu_dir は kifu_dir/*.sgf
            generate_supervised_learning_data_mt(program_dir=program_dir, kifu_dir=kifu_dir, board_size=size, opt=input_opt)
            # generate_supervised_learning_data(program_dir=program_dir, kifu_dir=kifu_dir, board_size=size, opt=input_opt)
            # generate_supervised_learning_data(program_dir=program_dir, kifu_dir=kifu_dir, board_size=size)

    if npz_dir is not None:###############
        if rl:
            if use_gpu:
                train_with_gumbel_alphazero_on_gpu(program_dir=program_dir, board_size=size, batch_size=BATCH_SIZE, rl_num=rl_num, rl_datetime=rl_datetime, network_name=network_name)
            else:
                train_with_gumbel_alphazero_on_cpu(program_dir=program_dir, board_size=size, batch_size=BATCH_SIZE)
        else:
            if use_gpu and ddp:
                train_on_gpu_ddp(program_dir=program_dir,board_size=size,  batch_size=BATCH_SIZE, epochs=EPOCHS, network_name=network_name, npz_dir=npz_dir, chckpoint_dir=checkpoint_dir)
            elif use_gpu:
                train_on_gpu(program_dir=program_dir,board_size=size,  batch_size=BATCH_SIZE, epochs=EPOCHS, network_name=network_name, npz_dir=npz_dir)
            else:
                train_on_cpu(program_dir=program_dir,board_size=size, batch_size=BATCH_SIZE, epochs=EPOCHS)



if __name__ == "__main__":
    train_main() # pylint: disable=E1120
