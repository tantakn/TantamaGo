# import json
# data = input()
# result = json.loads(data)
# print(f'Python got data and parsed {result["hoge"]}\n')



import os, sys, psutil, torch, threading, time, datetime, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from nn.utility import load_network, load_DualNet_128_12, choose_network
from nn.network.dual_net import DualNet
from monitoring import display_train_monitoring_worker

npz_dir = "/home/tantakn/code/TantamaGo/data/sl_data_0.npz"
# npz_dir = "/home0/y2024/u2424004/igo/TantamaGo/backup/data_Q50000/sl_data_0.npz"


model_file_path = "/home/tantakn/code/TantamaGo/model/sl-model_20241020_214243_Ep:14.bin"




BOARD_SIZE = 9
network_name1 = "DualNet"
use_gpu = True
gpu_num = 0
device = torch.device("cuda:0" if use_gpu else "cpu")

print("gpu_num: ", gpu_num)###############

# # ハードウェア使用率の監視スレッドを起動
# monitoring_worker = threading.Thread(target=display_train_monitoring_worker, args=(use_gpu, True, 10, os.getpid()), daemon=True)
# monitoring_worker.start()

network = choose_network(network_name1, model_file_path, use_gpu, gpu_num=gpu_num)

network.training = False

network.to(device)


def tmp_load_data_set(npz_path, rank=0):
    def check_memory_usage():
        if not psutil.virtual_memory().percent < 90:
            print(f"memory usage is too high. mem_use: {psutil.virtual_memory().percent}% [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
            assert True

    check_memory_usage()

    data = np.load(npz_path)

    check_memory_usage()

    plane_data = data["input"]
    policy_data = data["policy"].astype(np.float32)
    value_data = data["value"].astype(np.int64)

    check_memory_usage()

    plane_data = torch.tensor(plane_data)
    policy_data = torch.tensor(policy_data)
    value_data = torch.tensor(value_data)

    return plane_data, policy_data, value_data





data = "[[3,3,3,3,3,3,3,3,3,3,3],[3,1,1,1,1,0,1,0,1,0,3],[3,0,1,0,1,1,1,1,1,1,3],[3,1,1,1,1,1,1,1,1,1,3],[3,1,1,2,2,1,2,2,2,1,3],[3,1,2,0,2,2,2,2,1,1,3],[3,1,2,2,2,2,2,1,1,1,3],[3,0,1,2,0,2,2,1,2,1,3],[3,1,1,1,2,2,0,2,2,1,3],[3,1,0,1,2,2,2,0,2,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,0,0]]"
data = "[[3,3,3,3,3,3,3,3,3,3,3],[3,2,2,0,2,2,2,2,0,2,3],[3,2,0,2,2,0,2,2,1,2,3],[3,2,2,2,2,2,2,0,2,2,3],[3,2,2,2,2,2,2,2,2,1,3],[3,1,2,1,1,1,1,2,1,1,3],[3,1,1,1,0,1,1,1,1,1,3],[3,1,1,1,1,0,1,1,0,1,3],[3,1,1,0,1,1,0,1,1,1,3],[3,1,1,1,1,1,1,1,0,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,2,6]]"
"""
turn: 1
   1 2 3 4 5 6 7 8 9
 1 ● ┬ ● ● ┬ ● ● ● ● 
 2 ├ ● ● + ● ● ● ○ ┤ 
 3 ● ● ● ● ● ● + ● ○ 
 4 ● ● ● ● ● ● ● ○ ○ 
 5 ○ ○ ● ○ ● ● ● ● ● 
 6 ○ ○ ○ ○ ● + ● ● ● 
 7 ├ ○ + ○ ○ ● ● ● ┤ 
 8 ○ ○ ○ + ○ ● ● ● ● 
 9 ○ ○ ┴ ○ ● ● ● ┴ ● 
[[3,3,3,3,3,3,3,3,3,3,3],[3,1,0,1,1,0,1,1,1,1,3],[3,0,1,1,0,1,1,1,2,0,3],[3,1,1,1,1,1,1,0,1,2,3],[3,1,1,1,1,1,1,1,2,2,3],[3,2,2,1,2,1,1,1,1,1,3],[3,2,2,2,2,1,0,1,1,1,3],[3,0,2,0,2,2,1,1,1,0,3],[3,2,2,2,0,2,1,1,1,1,3],[3,2,2,0,2,1,1,1,0,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,4,8]]
[[[0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],[1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],[1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0]],[[1.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0,1.0],[0.0,1.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0,0.0,1.0,0.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0,1.0,1.0,1.0,1.0,1.0],[0.0,0.0,0.0,0.0,1.0,0.0,1.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0],[0.0,0.0,0.0,0.0,1.0,1.0,1.0,0.0,1.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0],[1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0],[1.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]]
2 9 2
"""
data = "[[3,3,3,3,3,3,3,3,3,3,3],[3,1,0,1,1,0,1,1,1,1,3],[3,0,1,1,0,1,1,1,2,0,3],[3,1,1,1,1,1,1,0,1,2,3],[3,1,1,1,1,1,1,1,2,2,3],[3,2,2,1,2,1,1,1,1,1,3],[3,2,2,2,2,1,0,1,1,1,3],[3,0,2,0,2,2,1,1,1,0,3],[3,2,2,2,0,2,1,1,1,1,3],[3,2,2,0,2,1,1,1,0,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,4,8]]"

"""turn: 1
   1 2 3 4 5 6 7 8 9
 1 ● ● ┬ ○ ○ ┬ ○ ○ ○ 
 2 ○ ○ ○ ○ ● + ○ ○ ○ 
 3 ○ ○ ○ + + ○ ○ ○ ● 
 4 ├ + + + + ○ ○ ● ● 
 5 ├ ○ + ○ ○ ○ ● + ● 
 6 ├ ○ ○ + ○ ○ ● ● ┤ 
 7 ○ + ○ ○ ○ ● ● ● ● 
 8 ● ○ ● ● ● + ● + ● 
 9 └ ○ ● ● ● ● ┴ ● ● 
[[3,3,3,3,3,3,3,3,3,3,3],[3,1,1,0,2,2,0,2,2,2,3],[3,2,2,2,2,1,0,2,2,2,3],[3,2,2,2,0,0,2,2,2,1,3],[3,0,0,0,0,0,2,2,1,1,3],[3,0,2,0,2,2,2,1,0,1,3],[3,0,2,2,0,2,2,1,1,0,3],[3,2,0,2,2,2,1,1,1,1,3],[3,1,2,1,1,1,0,1,0,1,3],[3,0,2,1,1,1,1,0,1,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,2,7]]
[[[0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0],[0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],[1.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,0.0],[1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0],[1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]],[[1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,1.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0],[1.0,0.0,1.0,1.0,1.0,0.0,1.0,0.0,1.0],[0.0,0.0,1.0,1.0,1.0,1.0,0.0,1.0,1.0]],[[0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0],[1.0,1.0,1.0,0.0,0.0,1.0,1.0,1.0,0.0],[0.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0],[0.0,1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0],[0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0],[1.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]],[[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]]]
4 3 2"""
data = "[[3,3,3,3,3,3,3,3,3,3,3],[3,1,1,0,2,2,0,2,2,2,3],[3,2,2,2,2,1,0,2,2,2,3],[3,2,2,2,0,0,2,2,2,1,3],[3,0,0,0,0,0,2,2,1,1,3],[3,0,2,0,2,2,2,1,0,1,3],[3,0,2,2,0,2,2,1,1,0,3],[3,2,0,2,2,2,1,1,1,1,3],[3,1,2,1,1,1,0,1,0,1,3],[3,0,2,1,1,1,1,0,1,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,2,7]]"


# data = input()
result = json.loads(data)

input_raw = json.loads(data)

input_np = np.zeros((6, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

for i in range(1, BOARD_SIZE + 1):
    for j in range(1, BOARD_SIZE + 1):
        if input_raw[i][j] == 0:
            input_np[0][i - 1][j - 1] = 1
        elif input_raw[i][j] == 1:
            input_np[1][i - 1][j - 1] = 1
        elif input_raw[i][j] == 2:
            input_np[2][i - 1][j - 1] = 1

color = input_raw[BOARD_SIZE + 2][0]
y = input_raw[BOARD_SIZE + 2][1]
x = input_raw[BOARD_SIZE + 2][2]

if y != 0:
    input_np[3][y - 1][x - 1] = 1
else:
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            input_np[4][i][j] = 1

if color == 1:
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            input_np[5][i][j] = 1
else:
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            input_np[5][i][j] = -1

print(input_np)######

# input_np, _, _ = tmp_load_data_set("/home/tantakn/code/TantamaGo/data/sl_data_0.npz") # npzファイルからデータをロード


input_planes = torch.tensor(input_np)
# print(input_planes)######

input_planes = input_planes.unsqueeze(0).to(device)  # バッチ次元を追加

# print(input_planes.shape)######
# print(input_planes)######

policy_data, value_data = network(input_planes)
print("network")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward(input_planes)
print("network.forward")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward_for_sl(input_planes)
print("network.forward_for_sl")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward_with_softmax(input_planes)
print("network.forward_with_softmax")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.inference(input_planes)
print("network.inference")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.inference_with_policy_logits(input_planes)
print("network.inference_with_policy_logits")
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)


# policy_data = policy_data.numpy()
# # print(np.sum(policy_data))######
# # policy_data = json.dumps(policy_data.tolist())
# # print(policy_data)######

# value_data = value_data.numpy()
# # print(np.sum(value_data))######
# # value_data = json.dumps(value_data.tolist())
# # print(value_data)######

# value_data = value_data.tolist()
# policy_data = policy_data.tolist()

# # s = "qwer"
# s = json.dumps(value_data)

# print(s, file=sys.stderr)

# print(s)

# data = {"policy": policy_data, "value": value_data}

# print(data, file=sys.stderr)

# data = json.dumps(data)


# print(data, file=sys.stderr)

# print(data)



# input_data, _, _ = tmp_load_data_set(npz_dir)

# input_plane = input_data[1234].unsqueeze(0)  # 1234番目の局面を抽出。バッチ次元を追加
# input_plane.to(device)

# print(input_plane.shape)######

# print(input_plane)######

# policy_data, value_data = network.inference(input_plane)

# print(policy_data)######
# print(value_data)######

# policy_data = policy_data.numpy()
# print(np.sum(policy_data))######
# policy_data = json.dumps(policy_data.tolist())
# print(policy_data)######

# value_data = value_data.numpy()
# print(np.sum(value_data))######
# value_data = json.dumps(value_data.tolist())
# print(value_data)######



# raw_policy, value_data = network.inference(input_planes)

# print(raw_policy)######
# print(value_data)######

# raw_policy, value_data = self.network.inference(input_planes)





# # import json
# # data = input()
# # result = json.loads(data)
# # print(f'Python got data and parsed {result["hoge"]}\n')



# """ とりあえずこれをinput、fowardする

# [[[1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]]]
#   """




# # def generate_input_planes(board: GoBoard, color: Stone, sym: int=0, opt: str="") -> np.ndarray:
# #     """ニューラルネットワークの入力データ（説明変数）を生成する。

# #     Args:
# #         board (GoBoard): 碁盤の情報。
# #         color (Stone): 手番の色。
# #         sym (int, optional): 対称形の指定. Defaults to 0.
# #         opt (str, optional): オプション。デフォルトは空文字列。 Defaults to "".

# #     Returns:
# #         numpy.ndarray: ニューラルネットワークの入力データ。
# #     """

# #     # 下の方に例がある。

# #     board_data = board.get_board_data(sym)
# #     """空点は0, 黒石は1, 白石は2の盤面の2次元リスト。"""
# #     board_size = board.get_board_size()
# #     # 手番が白の時は石の色を反転する.
# #     if color is Stone.WHITE:
# #         board_data = [datum if datum == 0 else (3 - datum) for datum in board_data]

# #     # 碁盤の各交点の状態
# #     #     空点 : 1枚目の入力面
# #     #     自分の石 : 2枚目の入力面
# #     #     相手の石 : 3枚目の入力面
# #     board_plane = np.identity(3)[board_data].transpose()

# #     # 直前の着手を取得
# #     _, previous_move, _ = board.record.get(board.moves - 1)

# #     # 直前の着手の座標
# #     #     着手 : 4枚目の入力面
# #     #     パス : 5枚目の入力面
# #     if board.moves > 1 and previous_move == PASS:
# #         history_plane = np.zeros(shape=(1, board_size ** 2))
# #         pass_plane = np.ones(shape=(1, board_size ** 2))
# #     else:
# #         previous_move_data = [1 if previous_move == board.get_symmetrical_coordinate(pos, sym) else 0 for pos in board.onboard_pos]
# #         history_plane = np.array(previous_move_data).reshape(1, board_size**2)
# #         pass_plane = np.zeros(shape=(1, board_size ** 2))

# #     # 手番の色 (6番目の入力面)
# #     # 黒番は1、白番は-1。現局面に打つ手番の色を示す。
# #     color_plane = np.ones(shape=(1, board_size**2))
# #     if color == Stone.WHITE:
# #         color_plane = color_plane * -1

# #     input_data = np.concatenate([board_plane, history_plane, pass_plane, color_plane]) \
# #         .reshape(6, board_size, board_size).astype(np.float32) # pylint: disable=E1121
    
# #     if opt == "semeai":
# #         semeai_data = board.get_liberty_data(sym)
# #         semeai_plane = [semeai_data[x:x+9] for x in range(0, board_size**2, board_size)]
# #         semeai_plane = np.array(semeai_plane).reshape(1, board_size**2)

# #         input_data = np.concatenate([board_plane, history_plane, pass_plane, color_plane, semeai_plane]) \
# #             .reshape(7, board_size, board_size).astype(np.float32) # pylint: disable=E1121

# #         return input_data
    
# #     # if opt == "semeai2":
# #         # TODO: 1眼、2眼、共通ダメ、ダメの情報を追加する。1目の1眼だけでなく複数目の1眼も判定したいけど難しそう？


# #     return input_data



# # def generate_target_data(board:GoBoard, target_pos: int, sym: int=0) -> np.ndarray:
# #     """教師あり学習で使用するターゲットデータ（目的変数）を生成する。

# #     Args:
# #         board (GoBoard): 碁盤の情報。
# #         target_pos (int): 教師データの着手の座標。
# #         sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

# #     Returns:
# #         np.ndarray: Policyのターゲットラベル。
# #     """
# #     target = [1 if target_pos == board.get_symmetrical_coordinate(pos, sym) else 0 for pos in board.onboard_pos]
# #     # パスだけ対称形から外れた末尾に挿入する。
# #     target.append(1 if target_pos == PASS else 0)
# #     #target_index = np.where(np.array(target) > 0)
# #     #return target_index[0]
# #     return np.array(target)


# # def generate_rl_target_data(board: GoBoard, improved_policy_data: str, sym: int=0) -> np.ndarray:
# #     """Gumbel AlphaZero方式の強化学習で使用するターゲットデータを精鋭する。

# #     Args:
# #         board (GoBoard): 碁盤の情報。
# #         improved_policy_data (str): Improved Policyのデータをまとめた文字列。
# #         sym (int, optional): 対称系の指定。値の範囲は0〜7の整数。デフォルトは0。

# #     Returns:
# #         np.ndarray: Policyのターゲットデータ。
# #     """
# #     split_data = improved_policy_data.split(" ")[1:]
# #     target_data = [1e-18] * len(board.board)

# #     for datum in split_data:
# #         pos, target = datum.split(":")
# #         coord = board.coordinate.convert_from_gtp_format(pos)
# #         target_data[coord] = float(target)

# #     target = [target_data[board.get_symmetrical_coordinate(pos, sym)] for pos in board.onboard_pos]
# #     target.append(target_data[PASS])

# #     return np.array(target)


# """
# この棋譜を読ませた
# (;FF[4]GM[1]SZ[9]
# AP[TantamaGo]PB[model/sl-model_default.bin]PW[model/sl-model_20240912_011228_Ep:14.bin]RE[B+R]KM[7.0];B[ed]C[82 A9:2.424e-07 B9:2.468e-07 C9:3.713e-07 D9:3.097e-07 E9:2.053...略

# GoGuiで保存したSGFファイルの中身。
# (;FF[4]CA[UTF-8]AP[GoGui:1.5.4]SZ[9]
# AB[bd][ce][cc][dc][eb][fh][ff][fe][fc][fb][gg][gf][gd][hd][hc]
# AW[be][cf][cd][df][dd][ef][ee][ed][ec][fg][fd][ge][gc][he]
# PL[W])
# """


# """
# 初期局面

#    A B C D E F G H J 
#  9 . . . . . . . . . 9 
#  8 . . . . . . . . . 8 
#  7 . . + . + . + . . 7 
#  6 . . . . . . . . . 6 
#  5 . . + . + . + . . 5 
#  4 . . . . . . . . . 4 
#  3 . . + . + . + . . 3 
#  2 . . . . . . . . . 2 
#  1 . . . . . . . . . 1 
#    A B C D E F G H J 
# White to play


# generate_input_planes:

# [[[1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

#  [[1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]
#   [1. 1. 1. 1. 1. 1. 1. 1. 1.]]]


# generate_target_data:

# [0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 1 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0
#  0]

# 分かりやすいように改行した。
# パスのとき10行目のが 1 になる


# 打った後の局面:

#    A B C D E F G H J 
#  9 . . . . . . . . . 9 
#  8 . . . . . . . . . 8 
#  7 . . + . + . + . . 7 
#  6 . . . . X . . . . 6 
#  5 . . + . + . + . . 5 
#  4 . . . . . . . . . 4 
#  3 . . + . + . + . . 3 
#  2 . . . . . . . . . 2 
#  1 . . . . . . . . . 1 
#    A B C D E F G H J 
# White to play
# """



# """
#    A B C D E F G H J 
#  9 . . . . . . . . . 9 
#  8 . . . . . . . . . 8 
#  7 . . + . + . + . . 7 
#  6 . . . . X . . . . 6 
#  5 . . X . + . + . . 5 
#  4 . . O . O X X . . 4 
#  3 . . + . + O + . . 3 
#  2 . . . . . . . . . 2 
#  1 . . . . . . . . . 1 
#    A B C D E F G H J 
# White to play

# G4に黒が伸びた局面。
# -1 があるのをプリントすると間隔が広がる。


# generate_input_planes:

# [[[ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#   [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#   [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#   [ 1.  1.  1.  1.  0.  1.  1.  1.  1.]
#   [ 1.  1.  0.  1.  1.  1.  1.  1.  1.]
#   [ 1.  1.  0.  1.  0.  0.  0.  1.  1.]
#   [ 1.  1.  1.  1.  1.  0.  1.  1.  1.]
#   [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]
#   [ 1.  1.  1.  1.  1.  1.  1.  1.  1.]]

#  [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  1.  0.  1.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  1.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  1.  0.  0.  0.  0.]
#   [ 0.  0.  1.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  1.  1.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  1.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]

#  [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
#   [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]

#  [[-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]
#   [-1. -1. -1. -1. -1. -1. -1. -1. -1.]]]


# generate_target_data:

# [0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 1 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0 0 0 0 0 0 0 0 0\
#  0]

# 分かりやすいように改行した。
# パスのとき10行目のが 1 になる



# D4に打った後の局面

#    A B C D E F G H J 
#  9 . . . . . . . . . 9 
#  8 . . . . . . . . . 8 
#  7 . . + . + . + . . 7 
#  6 . . . . X . . . . 6 
#  5 . . X . + . + . . 5 
#  4 . . O O O X X . . 4 
#  3 . . + . + O + . . 3 
#  2 . . . . . . . . . 2 
#  1 . . . . . . . . . 1 
#    A B C D E F G H J 
# Black to play
# """
