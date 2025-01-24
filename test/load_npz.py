import os, sys, psutil, torch, datetime, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


npz_dir = "/home/tantakn/code/TantamaGo/backup/kgs-19-2019-04/sl_data_0.npz"
# npz_dir = "/home/tantakn/code/TantamaGo/data/sl_data_3.npz"
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


def print_board_npz (a, b, dir, SIZE=9):
    """
    a: int npzのa~b番目のデータを見る
    b: int npzのa~b番目のデータを見る
    dir: str npzのディレクトリ
    SIZE: int 碁盤のサイズ
    """
    aaa, tmp, _ = tmp_load_data_set(dir)
    tmp = tmp.numpy()
    aaa = aaa.numpy()

    for i in range(a, b, 8):
        print("i: ", i)
        for j in range(SIZE):
            for k in range(SIZE):
                print(tmp[i][j * SIZE + k], end=" ")
            print()
        print(tmp[i][SIZE**2])
        for j in range(SIZE):
            for k in range(SIZE):
                if i / 8 % 2 == 0:
                    if (aaa[i][1][j][k] == 1):
                        print("●", end=" ")
                    elif (aaa[i][2][j][k] == 1):
                        print("○", end=" ")
                    else:
                        print("_", end=" ")
                else:
                    if (aaa[i][1][j][k] == 1):
                        print("○", end=" ")
                    elif (aaa[i][2][j][k] == 1):
                        print("●", end=" ")
                    else:
                        print("_", end=" ")
            print()
        print()



input_data, _, _ = tmp_load_data_set(npz_dir)

print(input_data.shape)######
# torch.Size([256000, 6, 9, 9])

input_plane = input_data[1234].unsqueeze(0)  # 1234番目の局面を抽出。バッチ次元を追加

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
for i in range(data.shape[2]):
    banmen.append([])
    for j in range(data.shape[3]):
        if data[0][0][i][j] == 1.0:
            banmen[i].append(0)
        elif data[0][1][i][j] == 1.0:
            banmen[i].append(1)
        else:
            banmen[i].append(2)

print(banmen)######
# [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0, 0], [0, 1, 2, 0, 0, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 2, 2, 1, 2, 2, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]

for x in banmen:
    for y in x:
        if y == 0:
            print("  ", end="")
        elif y == 1:
            print("● ", end="")
        else:
            print("○ ", end="")
    print()






    #         # 624
    #         # 1032
    #         # 1048
    #         # 1056
    #         # 1768
    #         # 2624
    #         # 2656
    #         # 3400
    #         # 3408
    #         # 3936
    #         # 3952
    #         # 3968
    #         # 3976
    #         # 4352
    #         # 4360
    #         # 4872
    #         # 4880
    #         # 7752
    #         # 7760
    #         # 8128
    #         # 8136
    #         # 9160
    #         # 9168
    #         # 9184
lis = [624, 1032, 1048, 1056, 1768, 2624, 2656, 3400, 3408, 3936, 3952, 3968, 3976, 4352, 4360, 4872, 4880, 7752, 7760, 8128, 8136, 9160, 9168, 9184]

for i in range(len(lis) - 1):
    print_board_npz(lis[i]- 40, lis[i] + 40, npz_dir, SIZE=19)