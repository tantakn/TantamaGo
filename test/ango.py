from cryptography.fernet import Fernet
import json

# キーを生成

# 任意の32バイトのバイト文字列を作成（例として32個の'a'を使用）
custom_key = b'a' * 32  # 必ず32バイトにします

# バイト文字列をBase64でエンコード
import base64
key = base64.urlsafe_b64encode(custom_key)

# key = Fernet.generate_key()
print(f"🐾key: {key}")

# Fernetオブジェクトを作成
f = Fernet(key)

# 暗号化する文字列
message = {
    "name": "TantamaGo",
    "age": 20
}

data = json.dumps(message)

# メッセージをバイト列に変換
message_bytes = data.encode()
print(f"🐾message_bytes: {message_bytes}")

# メッセージを暗号化
encrypted_message = f.encrypt(message_bytes)
print(f"🐾encrypted_message: {encrypted_message}")






# 復号化に使用するキー
# key = b'2rZjKmctpx1_Scc_bfJFOnmLjgNX9AJlXHlPsgxgQr8='

# Fernetオブジェクトを作成
f = Fernet(key)

# 復号化する文字列
# encrypted_message = b'gAAAAABhDpI1szEEj3kn5d_-0mfY5B5ux5E7q-iZmc8nCnLhHLNwGv_d_LaTXW8ouhKXn-SpALhfC7lT6A8kb6IRHmLj_gu4zQ=='

# メッセージを復号化
decrypted_message = f.decrypt(encrypted_message)

# 復号化したメッセージを文字列に変換
decrypted_message = decrypted_message.decode()

print(decrypted_message)