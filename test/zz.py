
import socket

# ソケットを作成
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# ホスト名とポート番号を設定
server_socket.bind(('localhost', 8000))
# 接続の待ち受け
server_socket.listen()

print('サーバーが起動しました。')

# クライアントからの接続を受け入れる
client_socket, addr = server_socket.accept()
print('クライアントと接続しました。')

# データを受信
data = client_socket.recv(1024).decode('utf-8')
print('受信したデータ:', data)

# ソケットを閉じる
client_socket.close()
server_socket.close()



import socket

# ソケットを作成
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# サーバーに接続
client_socket.connect(('localhost', 8000))



# データを送信
client_socket.send('こんにちは、サーバー！'.encode('utf-8'))

# ソケットを閉じる
client_socket.close()