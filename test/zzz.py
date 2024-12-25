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

t.cuda(0)
print(t)
