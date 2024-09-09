import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import logging
mylog = logging.getLogger("mylog")
mylog.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)



# import numpy as np
# print("qwer")

# def f (a):
#     return a + 3

# return [self.board[self.get_symmetrical_coordinate(pos, sym)].value for pos in self.onboard_pos]

# board = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])

# board_data = [0] * 9
# board_data = [1,1,1,1,1,1,1,1,1]
# board_data = [1,1,1,2,1,1,1,0,1]

# board_plane = np.identity(3)[board_data].transpose()

# print(board_plane)
# # for i in range (10):
# #     print(f(i))


# import datetime

# # 内側の二重引用符が外側の二重引用符と衝突しています。
# print(f"{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}")

# # 修正方法としては、内側の二重引用符を単一引用符 (') に変更することができます。
# print(f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")

# import re

# sgf = """
# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[]PW[]RE[B+10.0]KM[7.0];B[ha]C[82 A9:2.243e-10 B9:2.360e-10 C9:6.112e-02 D9:2.194e-10 E9:2.368e-10 F9:2.147e-10 G9:2.117e-10 H9:5.946e-02 J9:2.067e-10 A8:6.750e-02 B8:2.268e-10 C8:2.077e-10 D8:2.180e-10 E8:2.351e-10 F8:2.136e-10 G8:6.074e-02 H8:2.090e-10 J8:2.148e-10 A7:2.126e-10 B7:2.287e-10 C7:6.234e-02 D7:6.831e-02 E7:2.374e-10 F7:2.210e-10 G7:2.340e-10 H7:2.321e-10 J7:2.124e-10 A6:2.362e-10 B6:2.301e-10 C6:2.080e-10 D6:2.385e-10 E6:6.061e-02 F6:2.216e-10 G6:2.326e-10 H6:2.382e-10 J6:2.296e-10 A5:2.312e-10 B5:2.151e-10 C5:2.318e-10 D5:2.404e-10 E5:2.096e-10 F5:2.238e-10 G5:2.138e-10 H5:6.188e-02 J5:2.087e-10 A4:2.187e-10 B4:2.069e-10 C4:2.328e-10 D4:6.571e-02 E4:2.174e-10 F4:2.128e-10 G4:2.142e-10 H4:6.019e-02 J4:2.371e-10 A3:5.731e-02 B3:2.126e-10 C3:5.827e-02 D3:6.305e-02 E3:2.101e-10 F3:2.151e-10 G3:2.157e-10 H3:2.077e-10 J3:5.900e-02 A2:2.261e-10 B2:2.186e-10 C2:2.128e-10 D2:2.352e-10 E2:2.399e-10 F2:2.308e-10 G2:6.760e-02 H2:2.262e-10 J2:2.377e-10 A1:2.320e-10 B1:2.270e-10 C1:6.692e-02 D1:2.232e-10 E1:2.340e-10 F1:2.227e-10 G1:2.311e-10 H1:2.350e-10 J1:2.177e-10 pass:1.315e-10];W[eg]C[81 A9:2.220e-10 B9:2.117e-10 C9:6.371e-02 D9:2.069e-10 E9:2.333e-10 F9:2.247e-10 G9:2.213e-10 J9:2.281e-10 A8:2.261e-10 B8:2.402e-10 C8:2.149e-10 D8:2.233e-10 E8:2.240e-10 F8:6.169e-02 G8:2.177e-10 H8:2.203e-10 J8:2.343e-10 A7:6.073e-02 B7:2.310e-10 C7:2.
# """
# current_result = sgf.split('RE[')[1].split(']')[0]
# model1 = sgf.split('PB[')[1].split(']')[0]
# model2 = sgf.split('PW[')[1].split(']')[0]

# print(model1, model2)


import torch

# lis = [[1, 2], [3, 4]]
# lis2 = [[1, 2], [3, 4]]
# print(lis)
# [[1, 2], [3, 4]]

# ten = torch.tensor(lis)

# print(ten)
# # tensor([[1, 2],
# #         [3, 4]])

# nparr = np.array(lis)
# print(nparr)
# # [[1 2]
# #  [3 4]]

# ten2 = torch.ones_like(ten)

# nparr = np.array(lis)
# nparr2 = np.array(lis2)

# print(nparr * nparr2)
# print(np.sum(nparr))
# print(np.max(nparr))
# print(np.percentile(nparr, 25))

# narr = np.random.randint(0, 10, (2, 3))
# print(narr)
# # [[9 7 5]
# #  [7 9 7]]

# narr = np.random.randint(0, 10, (2, 3, 4, 5))
# print(narr)
# # [[[0 2 8 5]
# #   [7 7 6 1]
# #   [3 4 1 0]]

# #  [[2 5 7 9]
# #   [5 9 7 4]
# #   [4 8 8 1]]]

# print(np.sum(narr))
import datetime
print(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))