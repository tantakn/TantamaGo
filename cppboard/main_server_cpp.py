# (env) (base) u2424004@g14:~/igo/TantamaGo$ python3 main_server.py --password ****


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket, json, click, time, subprocess
from cryptography.fernet import Fernet


default_model_path = os.path.join("model_def", "sl-model_q50k_DualNet.bin")


@click.command()
@click.option('--password', type=click.STRING, help="パスワード。")
# @click.option('--ip', type=click.STRING, help="ip", default="0.0.0.0")
@click.option('--ip', type=click.STRING, help="ip", default="")
@click.option('--port', type=click.INT, help="port", default=51111)
def InetServer(password: str, ip: str="", port: int=51111):
    print("serverip: ", socket.gethostbyname(socket.gethostname()))

    if ip == "":
        ip = socket.gethostbyname(socket.gethostname())


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



    result = subprocess.run(['./goBoard'], capture_output=True, text=True)
    # print(result.stdout) # インスタンス.stdout で標準出力を取得できる

    try:
        while True:
            data = b''
            data = client_socket.recv(2048)

            if data != b'':
                print(f"受信データ「「\n{data}\n」」\n")#######
                # data = f.decrypt(data)
                data = data.decode()
                print(f'復号した受信データ「「\n{data}\n」」\n')#######
                if data == "exit" or data == "quit":
                    break

                output = client.run(data)
                if output is None:
                    output = "= \n"
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