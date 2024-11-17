import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket, json
from cryptography.fernet import Fernet


# ソケットを作成
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# サーバーに接続
client_socket.connect(('172.21.38.95', 51111))


# 送信するデータ
data = {
    "size": 9,
    "superko": True,
    "model": "mymodel"
}

# データ->json->バイト列に変換
data_json = json.dumps(data)

data_bytes = data_json.encode()


# メッセージを暗号化
my_key = "ttt"
for _ in range(32-len(my_key)):
    my_key += "0"
custom_key = my_key.encode()
import base64
key = base64.urlsafe_b64encode(custom_key)

f = Fernet(key)

encrypted_data = f.encrypt(data_bytes)
print(f"🐾encrypted_data: {encrypted_data}")


# メッセージを送信
client_socket.send(encrypted_data)


# ソケットを閉じる
client_socket.close()


# (envGo) PS C:\code\TantamaGo> & c:/code/TantamaGo/envGo/Scripts/python.exe c:/code/TantamaGo/test/t_cl.py
# 🐾encrypted_data: b'gAAAAABnNutONhqYpJ_Vs5QQH28AVjOpfkbsc6vUh8HocJrA7lVbriP-U6VyU_D3wvI-iL7qsdv4kLYkfZylTRa1w4cKB8OG62prmObZZoOTQCYBRU4ZlSA_ujFA-a8_FCe32YTMPBiu-Jw4OVlf4iWiYLhLZWvOgA=='