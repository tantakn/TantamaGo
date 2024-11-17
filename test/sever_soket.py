import socket  # ソケット通信のためのモジュールをインポート

# ブロッキングサーバーの基本クラスを定義
class BlockingServerBase:
    def __init__(self, timeout:int=60, buffer:int=1024):
        self.__socket = None  # サーバーソケットを初期化
        self.__timeout = timeout  # タイムアウト時間を設定
        self.__buffer = buffer  # 受信バッファサイズを設定
        self.close()  # 残留ソケットがある場合に備えてクローズ

    def __del__(self):
        self.close()  # オブジェクト削除時にソケットをクローズ

    def close(self) -> None:
        try:
            self.__socket.shutdown(socket.SHUT_RDWR)  # ソケットをシャットダウン
            self.__socket.close()  # ソケットをクローズ
        except:
            pass  # エラーが発生しても無視

    def accept(self, address, family:int, typ:int, proto:int) -> None:
        self.__socket = socket.socket(family, typ, proto)  # 新しいソケットを作成
        self.__socket.settimeout(self.__timeout)  # タイムアウトを設定
        self.__socket.bind(address)  # ソケットをアドレスにバインド
        self.__socket.listen(1)  # 接続待ちの状態にする。ここでの引数1はバックログと呼ばれ、接続待ちキューの最大数を指定します。つまり、同時に1つの接続要求を待機できることを意味します。
        print("Server started :", address)  # サーバー開始のメッセージを表示
        conn, _ = self.__socket.accept()  # クライアントからの接続を受け入れる

        while True:
            try:
                message_recv = conn.recv(self.__buffer).decode('utf-8')  # データを受信してデコード
                message_resp = self.respond(message_recv)  # 応答メッセージを生成
                conn.send(message_resp.encode('utf-8'))  # 応答メッセージをエンコードして送信
            except ConnectionResetError:
                break  # 接続がリセットされた場合はループを抜ける
            except BrokenPipeError:
                break  # パイプが壊れた場合はループを抜ける
        self.close()  # ソケットをクローズ

    def respond(self, message:str) -> str:
        return ""  # 応答メッセージを返す（継承先でオーバーライド）

# IPv4のサーバークラスを定義（BlockingServerBaseを継承）
class InetServer(BlockingServerBase):
    def __init__(self, host:str="127.0.1.1", port:int=8088) -> None:
        self.server=(host,port)  # サーバーのホストとポートを設定
        super().__init__(timeout=60, buffer=1024)  # 親クラスの初期化
        self.accept(self.server, socket.AF_INET, socket.SOCK_STREAM, 0)  # 接続を受け入れる

    def respond(self, message:str) -> str:
        print("received -> ", message)  # 受信したメッセージを表示
        return "Server accepted !!"  # 応答メッセージを返す

if __name__=="__main__":
    InetServer()  # サーバーを起動



    

# import socket


# class BlockingServerBase:
#     def __init__(self, timeout:int=60, buffer:int=1024):
#         self.__socket = None
#         self.__timeout = timeout
#         self.__buffer = buffer
#         self.close()

#     def __del__(self):
#         self.close()

#     def close(self) -> None:
#         try:
#             # 初回ではエラーが出るが、pass される
#             self.__socket.shutdown(socket.SHUT_RDWR)
#             # クラスのインスタンスが生成される際に、予期しない残留ソケットがある場合に備えて、close() メソッドを呼び出しています。
#             self.__socket.close()
#         except:
#             pass

#     def accept(self, address, family:int, typ:int, proto:int) -> None:
#         self.__socket = socket.socket(family, typ, proto)
#         self.__socket.settimeout(self.__timeout)
#         self.__socket.bind(address)
#         self.__socket.listen(1)
#         print("Server started :", address)
#         conn, _ = self.__socket.accept()

#         while True:
#             try:
#                 message_recv = conn.recv(self.__buffer).decode('utf-8')
#                 message_resp = self.respond(message_recv)
#                 conn.send(message_resp.encode('utf-8'))
#             except ConnectionResetError:
#                 break
#             except BrokenPipeError:
#                 break
#         self.close()

#     def respond(self, message:str) -> str:
#         return ""





# # 継承してる
# class InetServer(BlockingServerBase):
#     def __init__(self, host:str="172.21.38.95", port:int=8088) -> None:
#         self.server=(host,port)
#         super().__init__(timeout=60, buffer=1024)
#         self.accept(self.server, socket.AF_INET, socket.SOCK_STREAM, 0)

#     def respond(self, message:str) -> str:
#         print("received -> ", message)
#         return "Server accepted !!"



# if __name__=="__main__":
#     InetServer()