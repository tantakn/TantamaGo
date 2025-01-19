# サーバーを以下のように起動して。
# (env) (base) u2424004@g14:~/igo/TantamaGo$ python3 main_server.py --password ****
# serverip:  **サーバー側で表示された serverip のip**
# サーバーが起動しました。

# ローカルでこのコードを起動して、name コマンドで確認する。
# PS C:\code\TantamaGo> python .\main_client.py --password **** --ip **サーバー側で表示された serverip のip** --size 9 
# name
# = TantamaGo

# か、goguiの新規プログラムで
# python C:\code\TantamaGo\main_client.py --password **** --ip **サーバー側で表示された serverip のip** --size 9

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket, json
from cryptography.fernet import Fernet
import click
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from board.constant import BOARD_SIZE
from mcts.constant import NN_BATCH_SIZE, MCTS_TREE_SIZE
default_model_path = os.path.join("model", "model.bin")

@click.command()
@click.option('--password', type=click.STRING, help="パスワード。")
@click.option('--ip', type=click.STRING, help="ip", default="172.21.38.95")
@click.option('--port', type=click.INT, help="port", default=51111)
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤のサイズを指定。デフォルトは{BOARD_SIZE}。")
@click.option('--superko', type=click.BOOL, default=False, help="超劫の有効化フラグ。デフォルトはFalse。")
@click.option('--model', type=click.STRING, default="", \
    help=f"使用するニューラルネットワークのモデルパスを指定する。プログラムのホームディレクトリの相対パスで指定。\
    デフォルトは''。")
@click.option('--use-gpu', type=click.BOOL, default=False, \
    help="ニューラルネットワークの計算にGPUを使用するフラグ。デフォルトはFalse。")
@click.option('--policy-move', type=click.BOOL, default=False, \
    help="Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。")
@click.option('--sequential-halving', type=click.BOOL, default=False, \
    help="Gumbel AlphaZeroの探索手法で着手生成するフラグ。デフォルトはFalse。")
@click.option('--komi', type=click.FLOAT, default=7.0, \
    help="コミの値の設定。デフォルトは7.0。")
@click.option('--visits', type=click.INT, default=1000, \
    help="1手あたりの探索回数の指定。デフォルトは1000。\
    --const-timeオプション、または--timeオプションが指定された時は無視する。")
@click.option('--const-time', type=click.FLOAT, \
    help="1手あたりの探索時間の指定。--timeオプションが指定された時は無視する。")
@click.option('--time', type=click.FLOAT, \
    help="持ち時間の指定。")
@click.option('--batch-size', type=click.INT, default=-1, \
    help=f"探索時のミニバッチサイズ。デフォルトはNN_BATCH_SIZE = -1。")
@click.option('--tree-size', type=click.INT, default=-1, \
    help=f"探索木を構成するノードの最大数。デフォルトはMCTS_TREE_SIZE = -1。")
@click.option('--cgos-mode', type=click.BOOL, default=False, \
    help="全ての石を打ち上げるまでパスしないモード設定。デフォルトはFalse。")
@click.option('--net', type=click.STRING, default="DualNet", \
    help="--model のネットワーク。デフォルトは DualNet。DualNet_256_24 とかを指定する。")
def InerClient(password: str, size: int, superko: bool, model:str, use_gpu: bool, sequential_halving: bool, \
    policy_move: bool, komi: float, visits: int, const_time: float, time: float, \
    batch_size: int, tree_size: int, cgos_mode: bool, net: str, ip: str="", port: int=51111):
    """GTPクライアントの起動。

    Args:
        size (int): 碁盤の大きさ。
        superko (bool): 超劫の有効化フラグ。
        model (str): プログラムのホームディレクトリからのモデルファイルの相対パス。
        use_gpu (bool):  ニューラルネットワークでのGPU使用フラグ。デフォルトはFalse。
        policy_move (bool): Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。
        sequential_halving (bool): Gumbel AlphaZeroの探索手法で着手生成するフラグ。デフォルトはFalse。
        komi (float): コミの値。デフォルトは7.0。
        visits (int): 1手あたりの探索回数。デフォルトは1000。
        const_time (float): 1手あたりの探索時間。
        time (float): 対局時の持ち時間。
        batch_size (int): 探索実行時のニューラルネットワークのミニバッチサイズ。デフォルトはNN_BATCH_SIZE。
        tree_size (int): 探索木を構成するノードの最大数。デフォルトはMCTS_TREE_SIZE。
        cgos_mode (bool): 全ての石を打ち上げるまでパスしないモード設定。デフォルトはFalse。
    """

    key = password
    for _ in range(32-len(key)):
        key += "0"
    key = key.encode()
    import base64
    key = base64.urlsafe_b64encode(key)
    f = Fernet(key)

    # ソケットを作成
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.settimeout(300)

    # サーバーに接続
    client_socket.connect((ip, port))

    # 送信するデータ
    data = {
        "size": size,
        "superko": superko,
        "model": model,
        "use_gpu": use_gpu,
        "policy_move": policy_move,
        "sequential_halving": sequential_halving,
        "komi": komi,
        "visits": visits,
        "const_time": const_time,
        "time": time,
        "batch_size": batch_size,
        "tree_size": tree_size,
        "cgos_mode": cgos_mode,
        "net": net
    }

    data_json = json.dumps(data)
    data_bytes = data_json.encode()
    encrypted_data = f.encrypt(data_bytes)

    # print(f"🐾encrypted_data: {encrypted_data}")
    client_socket.send(encrypted_data)

    while True:
        data = input()

        data_bytes = data.encode()
        encrypted_data = f.encrypt(data_bytes)
        
        # print(f"🐾encrypted_data: {encrypted_data}")
        client_socket.send(encrypted_data)

        if data == "exit" or data == "quit":
            break

        data = client_socket.recv(1024)
        data = f.decrypt(data)
        data = data.decode()
        print(data)


    # ソケットを閉じる
    client_socket.close()

if __name__ == "__main__":
    InerClient()
