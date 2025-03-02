# (env) (base) u2424004@g14:~/igo/TantamaGo$ python3 main_server.py --password ****
# (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ /home/tantakn/code/TantamaGo/envGo/bin/python /home/tantakn/code/TantamaGo/main_server.py --use-gpu True --visits 1 --input-time 2000
# (envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo$ python3 main_server.py --use-gpu True --port 8000 --visits 10000 --input-time 100

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket, json, click, time
from cryptography.fernet import Fernet
from gtp.client import GtpClient
from gtp.client_socket import GtpClient_socket
from mcts.time_manager import TimeControl
from mcts.constant import NN_BATCH_SIZE, MCTS_TREE_SIZE
from board.constant import BOARD_SIZE


# default_model_path = os.path.join("model_def", "sl-model_20250227_033544_Ep00_13_1.bin")
default_model_path = os.path.join("model_def", "sl-model_q50k_DualNet.bin")



@click.command()
@click.option('--password', type=click.STRING, help="パスワード。")
@click.option('--ip', type=click.STRING, help="ip", default="0.0.0.0")
@click.option('--port', type=click.INT, help="port", default=8001)
@click.option('--size', type=click.IntRange(2, BOARD_SIZE), default=BOARD_SIZE, \
    help=f"碁盤のサイズを指定。デフォルトは{BOARD_SIZE}。")
@click.option('--superko', type=click.BOOL, default=False, help="超劫の有効化フラグ。デフォルトはFalse。")
@click.option('--model', type=click.STRING, default=default_model_path, \
    help=f"使用するニューラルネットワークのモデルパスを指定する。プログラムのホームディレクトリの相対パスで指定。\
    デフォルトは{default_model_path}。")
@click.option('--use-gpu', type=click.BOOL, default=False, \
    help="ニューラルネットワークの計算にGPUを使用するフラグ。デフォルトはFalse。")
@click.option('--policy-move', type=click.BOOL, default=False, \
    help="Policyの分布に従った着手生成処理フラグ。デフォルトはFalse。")
@click.option('--sequential-halving', type=click.BOOL, default=False, \
    help="Gumbel AlphaZeroの探索手法で着手生成するフラグ。デフォルトはFalse。")
@click.option('--komi', type=click.FLOAT, default=7.0, \
    help="コミの値の設定。デフォルトは7.0。")
@click.option('--visits', type=click.IntRange(min=1), default=1000, \
    help="1手あたりの探索回数の指定。デフォルトは1000。\
    --const-timeオプション、または--timeオプションが指定された時は無視する。")
@click.option('--const-time', type=click.FLOAT, \
    help="1手あたりの探索時間の指定。--timeオプションが指定された時は無視する。")
@click.option('--input-time', type=click.FLOAT, \
    help="持ち時間の指定。")
@click.option('--batch-size', type=click.IntRange(min=1), default=NN_BATCH_SIZE, \
    help=f"探索時のミニバッチサイズ。デフォルトはNN_BATCH_SIZE = {NN_BATCH_SIZE}。")
@click.option('--tree-size', type=click.IntRange(min=1), default=MCTS_TREE_SIZE, \
    help=f"探索木を構成するノードの最大数。デフォルトはMCTS_TREE_SIZE = {MCTS_TREE_SIZE}。")
@click.option('--cgos-mode', type=click.BOOL, default=False, \
    help="全ての石を打ち上げるまでパスしないモード設定。デフォルトはFalse。")
@click.option('--net', type=click.STRING, default="DualNet", \
    help="--model のネットワーク。デフォルトは DualNet。DualNet_256_24 とかを指定する。")
def InetServer(password: str, ip: str, port: int, size: int, superko: bool, model:str, use_gpu: bool, sequential_halving: bool, \
    policy_move: bool, komi: float, visits: int, const_time: float, input_time: float, \
    batch_size: int, tree_size: int, cgos_mode: bool, net: str):
    print("serverip: ", socket.gethostbyname(socket.gethostname()))

    # if ip == "":
    #     ip = socket.gethostbyname(socket.gethostname())



    # key = password
    # for _ in range(32-len(key)):
    #     key += "0"
    # key = key.encode()
    # import base64
    # key = base64.urlsafe_b64encode(key)
    # f = Fernet(key)


    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen()
    print('サーバーが起動しました。')
    client_socket, addr = server_socket.accept()
    print('クライアントと接続しました。')

    mode = TimeControl.CONSTANT_PLAYOUT
    if const_time is not None:
        mode = TimeControl.CONSTANT_TIME
    if time is not None:
        mode = TimeControl.TIME_CONTROL
    
    client = GtpClient_socket(BOARD_SIZE, True, default_model_path, use_gpu, policy_move, \
        sequential_halving, komi, mode, visits, const_time, input_time, batch_size, tree_size, \
        cgos_mode, net)

    is_gtp = True
    # is_gtp = False
    try:
        while True:
            data = b''
            # while True:
            #     msg = client_socket.recv(1024)
            #     print("msg: ", msg)#######
            #     if len(msg) <= 0:
            #         break
            #     data += msg
            #     print("data: ", data)#######
            #     msg = b''
            # print("test1")#######
            data = client_socket.recv(2048)

            if is_gtp == False and data != b'':
                print(f"受信データ（初期化中）「「\n{data}\n」」\n")#######
                # data = f.decrypt(data)
                data = data.decode()
                print(f"復号した受信データ（初期化中）「「\n{data}\n」」\n")#######

                data = json.loads(data)
                print("data['model']: ", data["model"])

                # size = data["size"]
                # superko = data["superko"]
                # model = data["model"]
                # use_gpu = data["use_gpu"]
                # policy_move = data["policy_move"]
                # sequential_halving = data["sequential_halving"]
                # komi = data["komi"]
                # visits = data["visits"]
                # const_time = data["const_time"]
                # time = data["time"]
                # batch_size = data["batch_size"]
                # tree_size = data["tree_size"]
                # cgos_mode = data["cgos_mode"]
                # net = data["net"]

                # if data["model"] == "":
                #     model = default_model_path
                # if data["batch_size"] == -1:
                #     batch_size = NN_BATCH_SIZE
                # if data["tree_size"] == -1:
                #     tree_size = MCTS_TREE_SIZE

                if data["model"] == "":
                    data["model"] = default_model_path
                if data["batch_size"] == -1:
                    data["batch_size"] = NN_BATCH_SIZE
                if data["tree_size"] == -1:
                    data["tree_size"] = MCTS_TREE_SIZE
                if data["net"] == "":
                    data["net"] = "DualNet"

                mode = TimeControl.CONSTANT_PLAYOUT

                if const_time is not None:
                    mode = TimeControl.CONSTANT_TIME
                if input_time is not None:
                    mode = TimeControl.TIME_CONTROL

                # if data["const_time"] is not None:
                #     mode = TimeControl.CONSTANT_TIME
                # if data["time"] is not None:
                #     mode = TimeControl.TIME_CONTROL

                program_dir = os.path.dirname(__file__)
                # client = GtpClient(data["size"], data["superko"], os.path.join(program_dir, data["model"]), data["use_gpu"], data["policy_move"], \
                # client = GtpClient(data["size"], data["superko"], os.path.join(program_dir, data["model"]), data["use_gpu"], data["policy_move"], \
                client = GtpClient_socket(data["size"], data["superko"], os.path.join(program_dir, data["model"]), data["use_gpu"], data["policy_move"], \
                    data["sequential_halving"], data["komi"], mode, visit, const_time, input_time, data["batch_size"], data["tree_size"], \
                    data["cgos_mode"], data["net"])
                # client = GtpClient_socket(data["size"], data["superko"], os.path.join(program_dir, data["model"]), data["use_gpu"], data["policy_move"], \
                #     data["sequential_halving"], data["komi"], mode, data["visits"], data["const_time"], data["time"], data["batch_size"], data["tree_size"], \
                #     data["cgos_mode"], data["net"])

                is_gtp = True

                continue

            if is_gtp == True and data != b'':
                print(f"受信データ「「\n{data}\n」」\n")#######
                # data = f.decrypt(data)
                data = data.decode()
                print(f'復号した受信データ「「\n{data}\n」」\n')#######
                if data == "exit" or data == "quit":
                    output = "= \n"
                    print(f"送信データ「「\n{output}\n」」\n")
                    output = output.encode()
                    # output = f.encrypt(output)
                    client_socket.send(output)
                    print(f"暗号化した送信データ「「\n{output}\n」」\n")
                    break

                output = client.run(data)
                if output is None:
                    output = "= \n"
                if output == "quit":
                    output = "= \n"
                    print(f"送信データ「「\n{output}\n」」\n")
                    output = output.encode()
                    # output = f.encrypt(output)
                    client_socket.send(output)
                    print(f"暗号化した送信データ「「\n{output}\n」」\n")
                    sys.exit(0)
                print(f"送信データ「「\n{output}\n」」\n")
                output = output.encode()
                # output = f.encrypt(output)
                client_socket.send(output)
                print(f"暗号化した送信データ「「\n{output}\n」」\n")
            
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("サーバーを終了します")
        client_socket.close()
        server_socket.close()
    finally:
        client_socket.close()
        server_socket.close()



    # # データを送信
    # client_socket.send('こんにちは、クライアント！'.encode('utf-8'))


    # ソケットを閉じる
    # クライアントとの通信が終了したら、client_socket.close()でソケットを適切に閉じ、リソースを解放することが重要です。
    client_socket.close()
    server_socket.close()


if __name__=="__main__":
    InetServer()  # サーバーを起動