import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket, json
from cryptography.fernet import Fernet

data = {
    "size": 9,
    "superko": True,
    "model": "mymodel"
}

print(data)

print(json.dumps(data))

