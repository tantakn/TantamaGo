import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
import psutil
import threading
import torch

n = np.array([[1, 2, 3], [4, 5, 6]])
print(n)
# [[1 2 3]
#  [4 5 6]]
print(n.shape)
# (2, 3)
print(n.dtype)
# int64

t = torch.tensor(n)
print(t)
# tensor([[1, 2, 3],
#         [4, 5, 6]])
print(t.shape)
# torch.Size([2, 3])
print(t.dtype)
# torch.int64




# def load_data_set(path: str):
#     """学習データセットを読み込む。シャッフルもする。

#     Args:
#         path (str): データセットのファイルパス。

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray]: 入力データ、Policy、Value。
#     """
#     data = np.load(path)

#     # それぞれの関係を保ったままシャッフルして返す
#     perm = np.random.permutation(len(data["value"]))
#     return data["input"][perm], \
#         data["policy"][perm].astype(np.float32), \
#         data["value"][perm].astype(np.int64)

# def myload(path, i):
#     import resource

#     soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
#     print(f"仮想メモリのソフトリミット: {soft_limit} バイト")
#     print(f"仮想メモリのハードリミット: {hard_limit} バイト")

#     data = np.load(path)
#     ttt = data["input"]
#     print(f"{i}: {ttt.__sizeof__()}\n", end="")
#     print(f"mem{i}: {psutil.virtual_memory()}\n", end="")

# if __name__ == "__main__":

#     import resource

#     soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
#     print(f"仮想メモリのソフトリミット: {soft_limit} バイト")
#     print(f"仮想メモリのハードリミット: {hard_limit} バイト")

#     print("mem: ", psutil.virtual_memory())

#     t1 = threading.Thread(target=myload, args=[0, "/home/tantakn/code/TantamaGo/data/sl_data_0.npz"])
#     t2 = threading.Thread(target=myload, args=[1, "/home/tantakn/code/TantamaGo/data/sl_data_1.npz"])
#     t3 = threading.Thread(target=myload, args=[2, "/home/tantakn/code/TantamaGo/data/sl_data_2.npz"])

#     t1.start()
#     t2.start()
#     t3.start()

#     data = np.load("/home/tantakn/code/TantamaGo/data/sl_data_0.npz")
#     ttt = data["input"]
#     print(ttt.__sizeof__())
#     print("mem: ", psutil.virtual_memory())

#     data1 = np.load("/home/tantakn/code/TantamaGo/data/sl_data_1.npz")
#     ttt1 = data["input"]
#     print(ttt1.__sizeof__())
#     print("mem1: ", psutil.virtual_memory())

#     data2 = np.load("/home/tantakn/code/TantamaGo/data/sl_data_2.npz")
#     ttt2 = data["input"]
#     print(ttt2.__sizeof__())
#     print("mem2: ", psutil.virtual_memory())

#     data3 = np.load("/home/tantakn/code/TantamaGo/data/sl_data_3.npz")
#     ttt3 = data["input"]
#     print(ttt3.__sizeof__())
#     print("mem3: ", psutil.virtual_memory())


#     te = [1,2,3,4,3,44,4]

#     print(te.__sizeof__())
