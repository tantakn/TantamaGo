import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket, json, click, time
from cryptography.fernet import Fernet
from gtp.client_socket import GtpClient_socket
from mcts.time_manager import TimeControl
from mcts.constant import NN_BATCH_SIZE, MCTS_TREE_SIZE


default_model_path = os.path.join("model_def", "sl-model_q50k_DualNet.bin")


@click.command()
@click.option('--password', type=click.STRING, help="パスワード。")
@click.option('--ip', type=click.STRING, help="ip", default="")
@click.option('--port', type=click.INT, help="port", default=51111)
def InetServer(password: str, ip: str="", port: int=51111):
    if ip == "":
        ip = socket.gethostbyname(socket.gethostname())


    key = password
    for _ in range(32-len(key)):
        key += "0"
    key = key.encode()
    import base64
    key = base64.urlsafe_b64encode(key)
    f = Fernet(key)


    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((ip, port))
    server_socket.listen()
    print('サーバーが起動しました。')
    client_socket, addr = server_socket.accept()
    print('クライアントと接続しました。')

    is_gtp = False
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
            data = client_socket.recv(1024)

            if is_gtp == False and data != b'':
                data = f.decrypt(data)
                data = data.decode()
                print('復号化したデータ:', data)

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

                mode = TimeControl.CONSTANT_PLAYOUT

                if data["const_time"] is not None:
                    mode = TimeControl.CONSTANT_TIME
                if data["time"] is not None:
                    mode = TimeControl.TIME_CONTROL

                program_dir = os.path.dirname(__file__)
                client = GtpClient_socket(data["size"], data["superko"], os.path.join(program_dir, data["model"]), data["use_gpu"], data["policy_move"], \
                    data["sequential_halving"], data["komi"], mode, data["visits"], data["const_time"], data["time"], data["batch_size"], data["tree_size"], \
                    data["cgos_mode"], data["net"])

                is_gtp = True

                continue

            if is_gtp == True and data != b'':
                print("data: ", data)#######
                data = f.decrypt(data)
                data = data.decode()
                print('復号化したデータ:', data)#######
                if data == "exit" or data == "quit":
                    break

                output = client.run(data)
                print("output: ", output)
                output = f.encrypt(output.encode())
                client_socket.send(output)
                print("output: ", output)
            
            time.sleep(1)

    except KeyboardInterrupt:
        print("サーバーを終了します")
        client_socket.close()
        server_socket.close()
    finally:
        client_socket.close()
        server_socket.close()



    # データを送信
    client_socket.send('こんにちは、クライアント！'.encode('utf-8'))


    # ソケットを閉じる
    # クライアントとの通信が終了したら、client_socket.close()でソケットを適切に閉じ、リソースを解放することが重要です。
    client_socket.close()
    server_socket.close()


if __name__=="__main__":
    InetServer()  # サーバーを起動