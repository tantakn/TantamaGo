"""自己対戦のエントリーポイント。
"""
import glob
import math
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
import click
from board.constant import BOARD_SIZE
from selfplay.worker import selfplay_worker, selfplay_worker_vs,  display_selfplay_progress_worker
from learning_param import SELF_PLAY_VISITS, NUM_SELF_PLAY_WORKERS, \
    NUM_SELF_PLAY_GAMES
from monitoring import display_train_monitoring_worker#############
import datetime

import torch
import multiprocessing

# pylint: disable=R0913, R0914
@click.command()
@click.option('--save-dir', type=click.STRING, default="archive", \
    help="棋譜ファイルを保存するディレクトリ。デフォルトはarchive。")
@click.option('--process', type=click.IntRange(min=1), default=NUM_SELF_PLAY_WORKERS, \
    help=f"自己対戦実行ワーカ数。デフォルトは{NUM_SELF_PLAY_WORKERS}。")
@click.option('--num-data', type=click.IntRange(min=1), default=NUM_SELF_PLAY_GAMES, \
    help="生成するデータ(棋譜)の数。デフォルトは10000。")
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤のサイズ。デフォルトは{BOARD_SIZE}。")
@click.option('--use-gpu', type=click.BOOL, default=True, \
    help="GPU使用フラグ。デフォルトはTrue。")
@click.option('--visits', type=click.IntRange(min=2), default=SELF_PLAY_VISITS, \
    help=f"自己対戦時の探索回数。デフォルトは{SELF_PLAY_VISITS}。")
@click.option('--model', type=click.STRING, default=os.path.join("model_def", "sl-model_default.bin"), \
    help="ニューラルネットワークのモデルファイルパス。デフォルトはmodelディレクトリ内のsl-model_default.bin。")
@click.option('--model2', type=click.STRING, default="None", \
    help="異なるモデルを対局させるときに指定する。")
@click.option('--net', 'network_name1', type=click.STRING, default="DualNet", \
    help="--model のネットワーク。デフォルトは DualNet。DualNet_256_24 とかを指定する。")
@click.option('--net2', 'network_name2', type=click.STRING, default="DualNet", \
    help="--model2 のネットワーク。デフォルトは DualNet。")
def selfplay_main(save_dir: str, process: int, num_data: int, size: int, use_gpu: bool, visits: int, model: str, model2: str, network_name1: str, network_name2: str):
    """自己対戦を実行する。

    Args:
        save_dir (str): 棋譜ファイルを保存するディレクトリ。デフォルトはarchive。
        process (int): 実行する自己対戦プロセス数。デフォルトは4。
        num_data (int): 生成するデータ数。デフォルトは10000。
        size (int): 碁盤のサイズ。デフォルトはBOARD_SIZE。
        use_gpu (bool): GPU使用フラグ。デフォルトはTrue
        visits (int): 自己対戦実行時の探索回数。デフォルトはSELF_PLAY_VISITS。
        model (str): 使用するモデルファイルのパス。デフォルトはmodel/model.bin。
        model2 (str): 使用するモデルファイルのパス。デフォルトはNone。
        network_name1 (str): 使用するニューラルネットワーク名。デフォルトはDualNet。
        network_name2 (str): 使用するニューラルネットワーク名。デフォルトはDualNet。
    """

    import resource

    # メモリ使用量を制限（単位：バイト）#########
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024 * 8, hard))  # n GBに制限
    print(f"🐾resource.getrlimit(resource.RLIMIT_AS): {resource.getrlimit(resource.RLIMIT_AS)}")###############

    monitoring_worker = threading.Thread(target=display_train_monitoring_worker, args=(use_gpu, True, 30, ), daemon=True)
    # monitoring_worker = threading.Thread(target=display_train_monitoring_worker, args=(use_gpu, True, 300, ), daemon=True)
    monitoring_worker.start()

    print("🐾model: ", model)#############
    print("🐾model2: ", model2)###############
    print("🐾network_name1: ", network_name1)###############
    print("🐾network_name2: ", network_name2)###############


    file_index_list = list(range(1, num_data + 1))
    split_size = math.ceil(num_data / process) # 切り上げ
    file_indice = [file_index_list[i:i+split_size] for i in range(0, len(file_index_list), split_size)] # 多分並行処理のために分けてる

    # num_data = 10, process = 4 なら
    # print(file_index_list)
    # # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # print(file_indice)
    # # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    # このコードは、指定されたディレクトリ save_dir 内のフォルダ名を取得し、その中で最大の数字を見つけて、新しいフォルダのインデックスを決定するためのものです。
    kifu_dir_index_list = [int(os.path.split(dir_path)[-1]) for dir_path in glob.glob(os.path.join(save_dir, "*"))]
    kifu_dir_index_list.append(0)
    kifu_dir_index = max(kifu_dir_index_list) + 1

    start_time = time.time()
    os.mkdir(os.path.join(save_dir, str(kifu_dir_index)))

    print(f"Self play visits : {visits}")

    # GPU番号を割り当てる
    cnt = 0
    def gpu_num():
        if not torch.cuda.is_available():
            return -1
        nonlocal cnt
        cnt += 1
        print("cnt: ", cnt)###################
        print("cnt % torch.cuda.device_count(): ", cnt % torch.cuda.device_count())###################
        return cnt % torch.cuda.device_count()

    if model2 == "None":
        # テンプレ改造？ここでsgfを出力してないselfplay_workerでしてる？
        # submit(selfplay_worker,...（selfplay_workerの引数たち）)らしい
        # max_workers=process は使用するプロセス数？
        with ProcessPoolExecutor(max_workers=process) as executor:
            futures = [executor.submit(selfplay_worker, os.path.join(save_dir, str(kifu_dir_index)), model, file_list, size, visits, use_gpu, network_name1, gpu_num=gpu_num()) for file_list in file_indice]

            monitoring_worker = threading.Thread(target=display_selfplay_progress_worker, args=(os.path.join(save_dir, str(kifu_dir_index)), num_data, use_gpu), daemon=True);
            monitoring_worker.start()

            # この .result() は結果を出力するのが目的ではなく、正常終了の確認が目的。
            # 多分、executor.shutdown(wait=True) でも良い。
            for future in futures:
                future.result()
    else:
        with ProcessPoolExecutor(max_workers=process) as executor:
            futures = [executor.submit(selfplay_worker_vs, os.path.join(save_dir, str(kifu_dir_index)), model, model2, file_list, size, visits, use_gpu, network_name1, network_name2, gpu_num=gpu_num()) for file_list in file_indice]

            monitoring_worker = threading.Thread(target=display_selfplay_progress_worker, args=(os.path.join(save_dir, str(kifu_dir_index)), num_data, use_gpu), daemon=True);
            monitoring_worker.start()

            for future in futures:
                future.result()

    finish_time = time.time() - start_time

# f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] generating\n

    print(f"[{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}] generating_finish\n{finish_time:3f} seconds, {(3600.0 * num_data / finish_time):3f} games/hour")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    selfplay_main() # pylint: disable=E1120