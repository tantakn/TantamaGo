import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket

# hostname = socket.gethostname()
# ip_address = socket.gethostbyname(hostname)
# print(f"サーバーのIPアドレス: {ip_address}")


# ソケットを作成
# socket.socket()関数を使用して、新しいソケットオブジェクトserver_socketを生成します。
# socket.AF_INETはIPv4アドレスファミリを指定します。
# socket.SOCK_STREAMはTCPプロトコルを使用することを指定し、信頼性の高いデータ転送を可能にします。
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# ホスト名とポート番号を設定
# bind()メソッドでサーバーのホスト名とポート番号を設定します。
# ('localhost', 8000)は自分自身のマシン（ループバックアドレス）でポート8000を利用することを意味します。
# これにより、サーバーは指定したアドレスとポートで接続を待ち受ける準備が整います。
# 多分、ホスト名≒IPアドレス≒住所、ポート番号≒ポストの番号？
server_socket.bind(('localhost', 8000))


# 接続の待ち受け
# listen()メソッドを呼び出して、サーバーを接続待ちの状態にします。
# 引数を指定しない場合、デフォルトの接続待ちキューサイズが適用されます。
# これにより、サーバーはクライアントからの接続要求を受け付けることができます。
# これしないと .accept() 出来なかった
# 多分、ポストを設置するみたいな役割。
server_socket.listen()
print('サーバーが起動しました。')


# クライアントからの接続を受け入れる
# server_socket.accept()メソッドを呼び出して、クライアントからの接続要求を受け付けています。
# このメソッドはブロッキングメソッドであり、クライアントからの接続があるまで処理を停止します。
# 接続が確立されると、新しいソケットオブジェクトclient_socketと、接続元のアドレスaddrが返されます。
# 以降、クライアントとの通信はclient_socketを通して行われます。
# クライアントの接続要求とは、多分クライアント側のコードの `client_socket.connect(('localhost', 8000))` のこと
# 多分、ポストに手紙が投函されるまで待って、届いたら接続できたという内容の手紙を返信する役割。最初の手紙には、相手側の情報が書いてある。
# 多分、client_socket 作ったら両方クライアントみたいに動かす
client_socket, addr = server_socket.accept()
print('クライアントと接続しました。')


# データを受信
# client_socket.recv(1024)を使用して、クライアントから最大1024バイトのデータを受信します。
# recv()メソッドは、指定したバイト数までのデータを受信するまでブロックされます。
# 受信したデータはバイト列（bytes型）となるため、decode('utf-8')でUTF-8エンコーディングの文字列（str型）にデコードしています。
# これにより、テキストデータとして処理が可能になります。
# 注意点
# recv()メソッドの引数は受信する最大バイト数を指定しますが、一度に全てのデータが受信されるとは限りません。そのため、大きなデータを扱う場合はループを使用して全てのデータを受信する実装が必要です。
# エンコーディングはクライアントとサーバーで一致させる必要があります。異なるエンコーディングを使用すると、デコード時にエラーが発生したり、文字化けの原因となります。
data = client_socket.recv(1024).decode('utf-8')
print('受信したデータ:', data)

import time
time.sleep(5)

client_socket.send('こんにちは、クライアント！'.encode('utf-8'))


# ソケットを閉じる
# クライアントとの通信が終了したら、client_socket.close()でソケットを適切に閉じ、リソースを解放することが重要です。
client_socket.close()
server_socket.close()