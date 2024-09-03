import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import logging
mylog = logging.getLogger("mylog")
mylog.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("🐕️%(asctime)s [🐾%(levelname)s🐾] %(pathname)s %(lineno)d %(funcName)s🐈️ %(message)s🦉", datefmt="%y%m%d_%H%M%S"))
mylog.addHandler(handler)


# @click.command()
# @click.option('--name', '-n', default='World')
# def cmd(name):
#     print(name)
#     print("BBB")

# def main():
#     cmd()

# print("aaaa")

# if __name__ == '__main__':
#     main()

# class tmpClass:
#     def __init__(self, initName): # クラスが宣言されたとき最初に実行する。self無いと動かない
#         self.name = initName # 多分普通はここでクラスの変数を宣言する？

#     def fn(self, ttt): # self無いと動かない
#         return self.name + ttt

#     age = 1000000 # クラスの変数はここでも宣言できる

#     def getAge(hoge): # 普通はself使うが、他の名前でも大丈夫
#         return hoge.age

#     def setAge(huga, tmpAge): # 普通はself使うが、他の名前でも大丈夫
#         huga.age = tmpAge

# t = tmpClass("qwer")
# print(t.fn("ASDF"))
# print(t.getAge())
# t.setAge(1234)
# print(t.getAge())

# ./data/text1.txt, ./data/text2.txt, ./data/text3.txt, ./data/text4.txt があったとして 

# import glob,re

# data_set = sorted(glob.glob("../**/*.py", recursive=True))

# for x in data_set:
#     if re.search()

# print(data_set)

# ->['./data/text1.txt', './data/text2.txt', './data/text3.txt', './data/text4.txt']


# lis = []
# for i in range(10):
#     lis += [i]
# print(lis[1:3])

# import math
# num_data = 100
# process = 4
# file_index_list = list(range(1, num_data + 1))
# split_size = math.ceil(num_data / process) # 多分並行処理のために分けてる
# file_indice = [file_index_list[i:i+split_size] \
#     for i in range(0, len(file_index_list), split_size)] # わからん

# print(file_index_list)
# print(file_indice)
# import glob
# search_dir = "foo"
# ttt = [dir_path for dir_path in glob.glob(os.path.join(search_dir, "*"))]
# print(ttt)



import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import glob

# # ./data/text2.txt, ./data/text3.txt, ./data/text1.txt, ./data/text4.txt があったとして。

# data_set = sorted(glob.glob(os.path.join("./", "SgfFile", "GoQuest_9x9_49893games", "sgf", "*.sgf")))
# data_set = sorted(glob.glob(os.path.join("./", "SgfFile", "GoQuest_9x9_49893games", "sgf", "*.sgf")))

# print(len(data_set))

# kifu_dir = "/data"

# kifu_index_list = [int(os.path.split(dir_path)[-1]) \
#                 for dir_path in glob.glob(os.path.join(kifu_dir, "*"))]

# # print(kifu_index_list)
# kifu_dir = "./SgfFile/20181218natsukaze_self/01"
# kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
# print(f"kifu_num: {kifu_num}")#############

# import datetime
# import time
# for i in range(1000):
#     time.sleep(10)
#     print(datetime.datetime.now())

import threading
import time
# def test(num):
#     print("aaa", num)
#     time.sleep(num)
#     print("bbb", num)

# for i in range(3):
#     t = threading.Thread(target=test, args=[i + 1],daemon = True)
#     t.start()

# print("ここで一度止まる")
# t.join(timeout=1)
# print("timeoutで動き出す")

# t.join()
# print("t 全部終わった")

# # これを使って、スレッドを一時停止させることができる。
# # グローバル変数。本当は引数として渡したほうが良いと思う。
# event = threading.Event()

# def thred_loop():
#     while(True):
#         # 初期状態ではここで止まる、event.set()で動き出す、event.clear()でまた止まる。
#         event.wait()
#         print("aaa")
#         time.sleep(1)

# def thred_watch():
#     while(True):
#         print(".", end="")
#         time.sleep(1)

# threading.Thread(target=thred_loop, daemon=True).start()
# threading.Thread(target=thred_watch, daemon=True).start()

# time.sleep(2)
# event.set()
# print("!set")
# time.sleep(2)
# event.clear()
# print("!clear")
# time.sleep(2)
# event.set()
# print("!set")
# time.sleep(2)
# event.clear()
# print("!clear")

# # ..!set
# # .aaa
# # aaa
# # .!clear
# # ..!set
# # aaa
# # .aaa
# # .!clear

# n = 5

# lis1 = list(range(n))
# print("lis1", lis1)

# lis2 = list(range(1, n + 1))
# print("lis2", lis2)

# lis3 = list()
# print("lis3", lis3)

# lis4 = []
# print("lis4", lis4)

# lis5 = [i for i in range(n)]
# print("lis5", lis5)

# # lis1 [0, 1, 2, 3, 4]
# # lis2 [1, 2, 3, 4, 5]
# # lis3 []
# # lis4 []
# # lis5 [0, 1, 2, 3, 4]


# import math 
# num_data = 10
# process = 4
# file_index_list = list(range(1, num_data + 1))
# split_size = math.ceil(num_data / process) # 切り上げ
# file_indice = [file_index_list[i:i+split_size] for i in range(0, len(file_index_list), split_size)] # 多分並行処理のために分けてる

# print(file_index_list)
# print(file_indice)

# # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# # [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

# import os
# print(os.path.join("aaa", "bbb", "ccc"))
# # aaa/bbb/ccc

# print(os.path.basename("aaa/bbb/ccc"))
# # ccc

# print(os.path.split("aaa/bbb/ccc"))
# # ('aaa/bbb', 'ccc')

# print([(os.path.split(dir_path)[-1]) for dir_path in glob.glob(os.path.join("testdir", "*"))])
# # ディレクトリのパスにあるファイルやフォルダのパスのリスト。指定したディレクトリのパスが絶対パスなら絶対パスで、相対パスなら相対パスで返ってくる。

# lis = [""] * 5
# print(lis)
# # ['', '', '', '', '']
# lis = [0] * 5
# print(lis)
# # [0, 0, 0, 0, 0]


# import click

# @click.command()
# @click.option('--name', '-n',type=click.STRING,  default='World', help="名前")
# @click.option('--num',type=click.IntRange(max=3),  default=1, help="繰り返す数")
# def foo(name, num):
#     for i in range(num):
#         print(f'Hello, {name}!')

# if __name__ == '__main__':
# #     foo()

# def func(a, b, c):
#     print(f'出力: a={a}, b={b}, c={c}')

# func(1, 2, 3)
# #=> 出力: a=1, b=2, c=3

# func(c=3, a=1, b=2)
# #=> 出力: a=1, b=2, c=3


# def func2(a, b, c=3):
#     print(f'出力: a={a}, b={b}, c={c}')

# func2(1, 2)
# #=> 出力: a=1, b=2, c=3

# func2(1, 2, 4)
# #=> 出力: a=1, b=2, c=4

# def func3(a: int, b: str, c: int = 3) -> None:
#     print(f'出力: a={a}, b={b}, c={c}')

# func3(1, 'qwer')
# #=> 出力: a=1, b=qwer, c=3

# func3('zxcv', 0.5, '4')
# #=> 出力: a=zxcv, b=0.5, c=4
# # これは型ヒントがあってもエラーにならない。

# import os

# a = ['1234', 'qwe', 'asdfg', 'zx']
# b = ['1234\n', 'qwe\n', 'asdfg\n', 'zx'] 

# with open("./test.txt", mode='w') as f:
#     f.writelines(a)

# with open("./test.txt", mode='r') as f:
#     print(f.read())

# #test.txt
# #1234qweasdfgzx

# with open("./test.txt", mode='w') as f:
#     f.writelines(b)

# with open("./test.txt", mode='r') as f:
#     print(f.read())


import re

pattern = re.compile(r'\d+')  # 数字にマッチするパターン

# match = pattern.match('123abc')
# if match:
#     print("Match found:", match.group())
# # Match found: 123

# match = pattern.match('abc123')
# if match:
#     print("Match found:", match.group())
# print(match)
# # None

# search = pattern.search('abc123')
# if search:
#     print("Search found:", search.group())
# # Search found: 123

# search = pattern.search('abc')
# if search:
#     print("Search found:", search.group())
# print(search)
# # None

# result = pattern.sub('NUMBER', 'abcdef')
# print("Substitution result:", result)
# # Substitution result: abcNUMBERdef

# split_result = pattern.split('abcdefghi')
# print("Split result:", split_result)
# # Split result: ['abc', 'def', 'ghi']

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


text = """
(;FF[4]GM[1]SZ[9]
AP[TantamaGo]PB[model/sl-model_default.bin_selfVs1]PW[model/sl-model_default.bin_selfVs2]RE[W+4.0]KM[7.0];B[hc]C[82 A9:2.259e-10 B9:2.377e-10 C9:2.151e-10 D9:2.210e-10 E9:2.385e-10 F9:2.162e-10 G9:2.132e-10 H9:2.114e-10 J9:2.082e-10 A8:2.375e-10 B8:6.412e-02 C8:2.
"""

# # print(re.findall(r"\[(.*?)\]", text))
# print(re.search(r"FF\[(.*?)\]", text).group(1))
# print(re.search(r"PB\[(.*?)\]", text).group(1))

# print(re.search(r"RE\[.*?\]", text).group(0))
# print(re.search(r"RE\[W\+88\.0\]", text).group(0))
# print(re.search(r"RE\[[BW0][\d.+]*?\]", text).group(0))
# print(re.search(r"RE\[([BW0])[.+\d]*?\]", text).group(1))


text2 = """
(;FF[4]GM[1]SZ[9]
AP[TantamaGo]PB[TantamaGo-Black]PW[TantamaGo-White]RE[W+R]KM[7.0];B[ee]C[82 A9:7.930e-07 B9:6.038e-07 C9:1.140e-06 D9:3.652e-06 E9:3.371e-06 F9:2.397e-06
"""

text3 = """
(;FF[4]GM[1]SZ[9]
AP[TantamaGo]PB[TantamaGo-Black]PW[TantamaGo-White]RE[W+-0.0]KM[7.0];B[ee]C[82 A9:7.271e-07 B9:5.
"""
# "B" or "W" or "0" (<-draw) ==
# win = re.search(r"RE\[([BW0])[.+\d]*?\]", sgf).group(1)
win = ""
try:
    print(re.search(r"RE\[[BW0]([\-.+\dR]*?)\]", text2).group(1))
    win = re.search(r"RE\[([BW0])[.+\-\dR]*?\]", text2).group(1) if re.search(r"RE\[[BW0]([\-.+\dR]*?)\]", text3).group(1) is not "+-0.0" else "0"
except Exception as e:
    mylog.debug(f"error: {e}")

print(win)