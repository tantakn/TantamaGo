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

# print(kifu_index_list)
kifu_dir = "./SgfFile/20181218natsukaze_self/01"
kifu_num = len(glob.glob(os.path.join(kifu_dir, "*.sgf")))######
print(f"kifu_num: {kifu_num}")#############