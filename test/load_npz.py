import os, sys, psutil, torch, datetime, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


npz_dir = "/home/tantakn/code/TantamaGo/data/sl_data_3.npz"
# npz_dir = "/home0/y2024/u2424004/igo/TantamaGo/backup/data_Q50000/sl_data_0.npz"


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


input_data, _, _ = tmp_load_data_set(npz_dir)

print(input_data.shape)######
# torch.Size([256000, 6, 9, 9])

input_plane = input_data[145].unsqueeze(0)  # 1234番目の局面を抽出。バッチ次元を追加

print(input_plane.shape)######
# torch.Size([1, 6, 9, 9])

print(input_plane)######
# tensor([[[[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
#           [ 1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
#           [ 1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.],
#           [ 1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.],
#           [ 1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
#           [ 1.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  1.],
#           [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
#           [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]],

#          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
#           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
#           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]],

#          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
#           [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

#          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

#          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

#          [[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])




data = input_plane.numpy()

data_json = json.dumps(data.tolist())

print(data_json)######
# [[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]]]



# 空点：0, 黒石：1, 白石：2
banmen = []
for i in range(9):
    banmen.append([])
    for j in range(9):
        if data[0][0][i][j] == 1.0:
            banmen[i].append(0)
        elif data[0][1][i][j] == 1.0:
            banmen[i].append(1)
        else:
            banmen[i].append(2)

print(banmen)######
# [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0, 0], [0, 1, 2, 0, 0, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 2, 2, 1, 2, 2, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]