import socket

hostname = socket.gethostname()
ip_address = socket.gethostbyname(hostname)
print(f"サーバーのIPアドレス: {ip_address}")