# 多分、c++のソケット通信は最初に name とかで往復する必要がある。

# クライアント側
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket, random, json, time


from board.constant import BOARD_SIZE
from mcts.constant import NN_BATCH_SIZE, MCTS_TREE_SIZE

secret = json.load(open(r"C:\code\TantamaGo\cppboard\gitignore_it.json"))
# secret = json.load(open("/home/tantakn/code/TantamaGo/cppboard/gitignor_it.json"))

is_sockt1_analyze = False
is_sockt2_analyze = False




# ソケットを作成
# client_socketというソケットオブジェクトを作成しています。
# socket.AF_INETはIPv4アドレスファミリを指定します。
# socket.SOCK_STREAMはTCPプロトコル（ストリームベースの通信）を指定します。
# これにより、IPv4のTCPソケットが生成されます。
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# サーバーに接続
# 接続先は'localhost'（自分自身のマシン）で、ポート番号は8000です。
# サーバー側でserver_socket.accept()が実行され、接続待ちの状態である必要があります。
client_socket1.connect((secret["ip_desk_ubuntu"], int(secret["port"])))
client_socket2.connect((secret["ip_desk_ubuntu"], int(secret["port2"])))


# データを送信
# サーバーにメッセージを送信しています。
# 'こんにちは、サーバー！'という文字列をencode('utf-8')でバイト列に変換します。
# client_socket.send()メソッドはバイト列を送信するため、エンコードが必要です。
client_socket1.send('name'.encode('utf-8'))
client_socket2.send('name'.encode('utf-8'))


# データを受信
data = client_socket1.recv(1024).decode('utf-8')
data = client_socket2.recv(1024).decode('utf-8')
# print('受信したデータ:', data)





# # tamagoならここから
# data = {
#     "size": BOARD_SIZE,
#     "superko": "Ture",
#     "model": "/home/tantakn/code/TantamaGo/model_def/sl-model_q50k_DualNet_256_24.bin",
#     # "model": "/home/tantakn/code/TantamaGo/model_def/sl-model_20250227_033544_Ep00_13_1.bin",
#     "use_gpu": "True",
#     "policy_move": "False",
#     "sequential_halving": "False",
#     "komi": 7,
#     "visits": 10000,
#     "const_time": 10,
#     "time": None,
#     "batch_size": -1,
#     "tree_size": -1,
#     "cgos_mode": "False",
#     "net": "DualNet_256_24"
# }

# data_json = json.dumps(data)
# data_bytes = data_json.encode()
# # encrypted_data = f.encrypt(data_bytes)

# # print(f"🐾encrypted_data: {encrypted_data}")
# client_socket1.send(data_bytes)
# client_socket2.send(data_bytes)
# # ここまで必要



GPTALPHABET = "ABCDEFGHJKLMNOPQRST"
GPTAlapabet = "abcdefghjklmnopqrst"

def convert_int_to_gpt(num):
    return GPTALPHABET[num]

def convert_gpt_to_int(c):
    if GPTAlapabet.find(c) != -1:
        return GPTAlapabet.find(c)
    elif GPTALPHABET.find(c) != -1:
        return GPTALPHABET.find(c)
    else :
        assert(False)

def Is_black(c):
    if c == "B" or c == "b" or c == "black" or c == "Black" or c == "BLACK":
        return True
    else:
        return False

def Is_white(c):
    if c == "W" or c == "w" or c == "white" or c == "White" or c == "WHITE":
        return True
    else:
        return False

def Is_pass(c):
    if c == "pass" or c == "Pass" or c == "PASS":
        return True
    else:
        return False


n = 0
while True:
    s = input()

    if s == 'exit':
        client_socket1.send(s.encode('utf-8'))
        client_socket2.send(s.encode('utf-8'))
        break

    # print('送信するデータ:', s)

    # s = "qqwer"

    inputs = s.split(' ')

    if inputs[0] == "genmove":
        if not Is_black(inputs[1]):
            if is_sockt1_analyze:
                s = "lz-genmove_analyze " + inputs[1]
            client_socket1.send(s.encode('utf-8'))
            # print(f"socket1_send[{s}]")##########
            data1 = client_socket1.recv(1024).decode('utf-8')
            # print(f"socet1_recv[{data1}]")##########

            client_socket2.send(f"play {inputs[1]} {data1.split('=')[1] if not data1.split('=')[1][0] == ' ' else data1.split('=')[1].split(' ')[1]}".encode('utf-8'))
            # print(f"socket2_send[play B {data1.split('=')[1]}]")##########
            data2 = client_socket2.recv(1024).decode('utf-8')
            # print(f"socket2_recv[{data2}]")############

            # assert(data2 == "=\n")
            print(data1)
        else:
            if is_sockt2_analyze:
                s = "lz-genmove_analyze " + inputs[1]
            client_socket2.send(s.encode('utf-8'))
            # print(f"socket2_send[{s}]")##############
            data2 = client_socket2.recv(1024).decode('utf-8')
            # print(f"socket2_recv[{data2}]")############

            client_socket1.send(f"play {inputs[1]} {data2.split('=')[1] if not data2.split('=')[1][0] == ' ' else data2.split('=')[1].split(' ')[1]}".encode('utf-8'))
            # print(f"socket1_send[play W {data2.split('=')[1].split(' ')[1].split('\n')[0]}]")############
            data1 = client_socket1.recv(1024).decode('utf-8')
            # print(f"socket1_recv[{data1}]")###########

            # assert(data1 == "=\n")
            print(data2)
    else:
            client_socket1.send(s.encode('utf-8'))
            # print(f"socket1_send[{s}]")##########
            client_socket2.send(s.encode('utf-8'))
            # print(f"socket2_send[{s}]")###############

            data1 = client_socket1.recv(1024).decode('utf-8')
            # print(f"socket1_recv[{data1}]")##############
            data2 = client_socket2.recv(1024).decode('utf-8')
            # print(f"socket2_recv[{data2}]")###############

            print(data2)




    # client_socket.send(s.encode('utf-8'))
    # data = client_socket.recv(1024).decode('utf-8')
    # print(data)

    time.sleep(0.1)


# ソケットを閉じる
client_socket1.close()
client_socket2.close()