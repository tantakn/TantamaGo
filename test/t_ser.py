import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket, json
from cryptography.fernet import Fernet


# ソケットを作成
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('172.21.38.95', 51111))


# 接続の待ち受け
server_socket.listen()
print('サーバーが起動しました。')


# クライアントからの接続を受け入れる
client_socket, addr = server_socket.accept()
print('クライアントと接続しました。')


# データを受信
# 暗号データの復号はバイト列で行うから client_socket.recv(1024).decode('utf-8') の .decode('utf-8') は不要
encrypted_data = client_socket.recv(1024)
print('受信したデータ:', encrypted_data)


# 復号化
my_key = "keytest"
for _ in range(32-len(my_key)):
    my_key += "0"
custom_key = my_key.encode()
import base64
key = base64.urlsafe_b64encode(custom_key)

f = Fernet(key)

# 復号
decrypted_message = f.decrypt(encrypted_data)

# バイト列を文字列（json）に変換
decrypted_message = decrypted_message.decode()
print('復号化したデータ:', decrypted_message)

# json から dict に変換
data = json.loads(decrypted_message)
print("data: ", data)
print("data['model']: ", data['model'])


# ソケットを閉じる
client_socket.close()
server_socket.close()


# (base) u2424004@g14:~$ /bin/python3 /data/student/u2424004/igo/TantamaGo/test/t_ser.py
# サーバーが起動しました。
# クライアントと接続しました。
# 受信したデータ: b'gAAAAABnNutONhqYpJ_Vs5QQH28AVjOpfkbsc6vUh8HocJrA7lVbriP-U6VyU_D3wvI-iL7qsdv4kLYkfZylTRa1w4cKB8OG62prmObZZoOTQCYBRU4ZlSA_ujFA-a8_FCe32YTMPBiu-Jw4OVlf4iWiYLhLZWvOgA=='
# 復号化したデータ: {"size": 9, "superko": true, "model": "mymodel"}
# data:  {'size': 9, 'superko': True, 'model': 'mymodel'}
# data['model']:  mymodel