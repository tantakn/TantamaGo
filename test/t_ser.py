import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import socket
from cryptography.fernet import Fernet
import click


@click.command()
@click.option('--password', type=click.STRING, help="パスワード。")
@click.option('--ip', type=click.STRING, help="ip", default="")
@click.option('--port', type=click.INT, help="port", default=51111)
def InetServer(password: str, ip: str="", port: int=51111):
    if ip == "":
        ip = socket.gethostbyname(socket.gethostname())

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    server_socket.bind((ip, port))


    server_socket.listen()
    print('サーバーが起動しました。')


    client_socket, addr = server_socket.accept()
    print('クライアントと接続しました。')


    data = client_socket.recv(1024)
    print('受信したデータ:', data)


    for _ in range(32-len(password)):
        my_key += "0"
    tmp_key = my_key.encode()
    import base64
    key = base64.urlsafe_b64encode(tmp_key)

    f = Fernet(key)

    # メッセージを復号化
    decrypted_message = f.decrypt(data)

    # 復号化したメッセージを文字列に変換
    decrypted_message = decrypted_message.decode()
    print('復号化したデータ:', decrypted_message)

    import json
    data = {
        "size": "",
        "superko": True,
        "model": ""
    }
    data = json.loads(decrypted_message)
    print("data[model]: ", data[model])

    import time
    time.sleep(5)

    # データを送信
    client_socket.send('こんにちは、クライアント！'.encode('utf-8'))


    # ソケットを閉じる
    # クライアントとの通信が終了したら、client_socket.close()でソケットを適切に閉じ、リソースを解放することが重要です。
    client_socket.close()
    server_socket.close()


if __name__=="__main__":
    InetServer()  # サーバーを起動