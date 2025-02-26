# クライアント側
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket, random, json, time

secret = json.load(open("gitignore_it.json"))


# ソケットを作成
# client_socketというソケットオブジェクトを作成しています。
# socket.AF_INETはIPv4アドレスファミリを指定します。
# socket.SOCK_STREAMはTCPプロトコル（ストリームベースの通信）を指定します。
# これにより、IPv4のTCPソケットが生成されます。
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# サーバーに接続
# 接続先は'localhost'（自分自身のマシン）で、ポート番号は8000です。
# サーバー側でserver_socket.accept()が実行され、接続待ちの状態である必要があります。
client_socket.connect((secret["ip_desk_ubuntu"], secret["port"]))


# データを送信
# サーバーにメッセージを送信しています。
# 'こんにちは、サーバー！'という文字列をencode('utf-8')でバイト列に変換します。
# client_socket.send()メソッドはバイト列を送信するため、エンコードが必要です。
client_socket.send('name'.encode('utf-8'))


# データを受信
data = client_socket.recv(1024).decode('utf-8')
# print('受信したデータ:', data)

n = 0
while True:
    s = input()

    if s == 'exit':
        client_socket.send(s.encode('utf-8'))
        break

    # print('送信するデータ:', s)

    # s = "qqwer"

    client_socket.send(s.encode('utf-8'))
    data = client_socket.recv(1024).decode('utf-8')
    print(data)

    time.sleep(0.1)


# ソケットを閉じる
client_socket.close()