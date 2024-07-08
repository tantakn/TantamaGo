import os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))


import click


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

class tmpClass:
    def __init__(self, initName): # self無いと動かない
        self.name = initName

    def aaa(self, t):
        return self.name + t
    
t = tmpClass("qwer")
print(t.aaa("zxcv"))

# import glob

# data_set = sorted(glob.glob(os.path.join("./", "data", "sl_data_*.npz")))

# print(data_set)