import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import logging
mylog = logging.getLogger("mylog")
mylog.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)




import re

# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[model/sl-model_default.bin-Black]PW[model/sl-model20240711.bin-White]RE[W+88.0]KM[7.0];B[ha]C[82 A9:2.243e-10 B9:2.360e-10 C9:6.112e-02 D9:2.

# with open("./archive/6/1.sgf") as f:
#     sgf = f.read()
#     model1 = re.search(r"PB\[(.*)\]PW", sgf)
#     if model1 is not None:
#         print(model1.group(1))
#     else:
#         print("None")


# text = "<<<hoge>>>>>>"
# pattern = r'<*([^<>]+)>*'
# print(re.search(pattern, text).group(1))
# # hoge


# text = """
# # (;FF[4]GM[1]SZ[9]
# # AP[TantamaGo]PB[model/sl-model_default.bin-Black]PW[model/sl-model20240711.bin-White]RE[W+88.0]KM[7.0];B[ha]C[82 A9:2.243e-10 B9:2.360e-10 C9:6.112e-02 D9:2.
# """

# pattern = re.compile(r'.*?\[(.*?)\].*?\[(.*?)\]')

# matches = pattern.match(text)
# print(matches.groups())


# pattern = re.compile(r'\[.*?\]')

# matches = pattern.findall(text)
# for match in matches:
#     print("Matched:", match)

# text = "<qwer><asd>"
# ptn = r'<(.*?)>'
# print(re.search(ptn, text).groups())
# # ('qwer',)
# print(re.findall(ptn, text))
# # ['qwer', 'asd']

# text = "<qwer><asd>"
# ptn = r'<(.*?)><(.*?)>'
# print(re.search(ptn, text).groups())
# # ('qwer',)
# print(re.findall(ptn, text))
# # ['qwer', 'asd']



# text = """
# qwert
# sdfgh
# """

# print(re.search(r"w[\s\S]*g", text).group() if re.search(r"w[\s\S]*g", text) else "None")
# # wert
# # sdfg
# print(re.search(r"w.*g", text, flags=re.DOTALL).group())
# # wert
# # sdfg
# lis = [[0] * 3 for i in range(3)]

# print(lis)
# #->[[0, 0, 0], [0, 0, 0], [0, 0, 0]]

# lis[1][2] = 1

# print(lis)
# #->[[0, 0, 0], [0, 0, 1], [0, 0, 0]]


# t1 = """
# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[model/sl-model_default.bin_selfVs1]PW[model/sl-model_default.bin_selfVs2]RE[W+4.0]KM[7.0];B[hc]C[82 A9:2.259e-
# """

# t2 = """
# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[TantamaGo-Black]PW[TantamaGo-White]RE[W+R]KM[7.0];B[ee]C[82 A9:7.930e-07 B9:6.038e-07 C9:1.140e-06 D9:3.652e-06 E9:3.371e-06 F9:2.397e-06
# """

# t3 = """
# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[TantamaGo-Black]PW[TantamaGo-White]RE[W+-0.0]KM[7.0];B[ee]C[82 A9:7.271e-07 B9:5.
# """

# print(text.split('RE[')[1].split(']')[0])


# import subprocess

# # with subprocess.

# process = subprocess.run()


print(os.path.exists("zzlog"))
print(os.path.isdir("zzlog"))
print(os.path.isdir("zlog"))