import os, shutil
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))



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

import os
print(os.path.join("aaa", "bbb", "ccc"))
# aaa/bbb/ccc

print(os.path.basename("aaa/bbb/ccc"))
# ccc

print(os.path.split("aaa/bbb/ccc"))
# ('aaa/bbb', 'ccc')

print([(os.path.split(dir_path)[-1]) for dir_path in glob.glob(os.path.join("testdir", "*"))])
# ディレクトリのパスにあるファイルやフォルダのパスのリスト。指定したディレクトリのパスが絶対パスなら絶対パスで、相対パスなら相対パスで返ってくる。