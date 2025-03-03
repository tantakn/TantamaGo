import os, sys, psutil, torch, datetime, json
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np


# npz_dir = "/home/tantakn/code/TantamaGo/backup/kgs-19-2019-04/sl_data_0.npz"
# npz_dir = "/home/tantakn/code/TantamaGo/backup/data_Q50000/sl_data_0.npz"
npz_dir = "/home/tantakn/code/TantamaGo/data/sl_data_0.npz"
# npz_dir = "/home0/y2024/u2424004/igo/TantamaGo/backup/data_Q50000/sl_data_0.npz"
npz_dir = "/home0/y2024/u2424004/igo/TantamaGo/data/sl_data_500.npz"


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


def print_board_npz (a, b, c, dir, SIZE=9):
    """
    a: int npzのa~b番目のデータc飛ばしでを見る
    b: int npzのa~b番目のデータc飛ばしでを見る
    c: int npzのa~b番目のデータc飛ばしでを見る
    dir: str npzのディレクトリ
    SIZE: int 碁盤のサイズ
    """
    inputs, policys, values = tmp_load_data_set(dir)
    inputs = inputs.numpy()
    policys = policys.numpy()
    values = values.numpy()

    for i in range(a, b, 8 * c):
        print("i: ", i)
        print("teban(黒1, 白-1): ", inputs[i][5][0][0])
        for j in range(SIZE):
            for k in range(SIZE):
                if i / 8 % 2 == 0:
                    if (inputs[i][1][j][k] == 1):
                        print("●", end=" ")
                    elif (inputs[i][2][j][k] == 1):
                        print("○", end=" ")
                    else:
                        print("_", end=" ")
                else:
                    if (inputs[i][1][j][k] == 1):
                        print("○", end=" ")
                    elif (inputs[i][2][j][k] == 1):
                        print("●", end=" ")
                    else:
                        print("_", end=" ")
            print()
        print()

        print("target")
        for j in range(SIZE):
            for k in range(SIZE):
                print(policys[i][j * SIZE + k], end=" ")
            print()
        print("pass: ", policys[i][SIZE**2])

        print("value(手番が勝つとき2): ", values[i])

        print("json")
        # 空点：0, 黒石：1, 白石：2
        banmen = []
        for j in range(inputs.shape[2]):
            banmen.append([])
            for k in range(inputs.shape[3]):
                if inputs[i][0][j][k] == 1.0:
                    banmen[j].append(0)
                elif inputs[i][1][j][k] == 1.0:
                    banmen[j].append(1)
                else:
                    banmen[j].append(2)

        print(banmen)######
        # [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0, 0], [0, 1, 2, 0, 0, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 2, 2, 1, 2, 2, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]



# input_data, _, _ = tmp_load_data_set(npz_dir)

# print(input_data.shape)######
# # torch.Size([256000, 6, 9, 9])

# input_plane = input_data[1234].unsqueeze(0)  # 1234番目の局面を抽出。バッチ次元を追加

# print(input_plane.shape)######
# # torch.Size([1, 6, 9, 9])

# print(input_plane)######
# # tensor([[[[ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
# #           [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
# #           [ 1.,  0.,  1.,  1.,  1.,  1.,  0.,  1.,  1.],
# #           [ 1.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  1.],
# #           [ 1.,  1.,  1.,  0.,  1.,  0.,  1.,  0.,  1.],
# #           [ 1.,  1.,  0.,  0.,  0.,  1.,  1.,  1.,  1.],
# #           [ 1.,  1.,  0.,  1.,  1.,  0.,  0.,  1.,  1.],
# #           [ 1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  1.],
# #           [ 1.,  1.,  1.,  1.,  1.,  1.,  0.,  1.,  1.]],

# #          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.],
# #           [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
# #           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  1.,  0.,  0.,  1.,  1.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]],

# #          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.],
# #           [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  1.,  1.,  0.,  1.,  1.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

# #          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

# #          [[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
# #           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]],

# #          [[-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.],
# #           [-1., -1., -1., -1., -1., -1., -1., -1., -1.]]]])




# input_data = input_plane.numpy()

# input_data_json = json.dumps(input_data.tolist())

# print(input_data_json)######
# # [[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]]]]



# # 空点：0, 黒石：1, 白石：2
# banmen = []
# for i in range(input_data.shape[2]):
#     banmen.append([])
#     for j in range(input_data.shape[3]):
#         if input_data[0][0][i][j] == 1.0:
#             banmen[i].append(0)
#         elif input_data[0][1][i][j] == 1.0:
#             banmen[i].append(1)
#         else:
#             banmen[i].append(2)

# print(banmen)######
# # [[0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 2, 0, 0], [0, 1, 2, 0, 0, 0, 2, 1, 0], [0, 0, 0, 1, 0, 2, 0, 1, 0], [0, 0, 1, 2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1, 1, 0, 0], [0, 0, 0, 2, 2, 1, 2, 2, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0]]

# for x in banmen:
#     for y in x:
#         if y == 0:
#             print("_ ", end="")
#         elif y == 1:
#             print("● ", end="")
#         else:
#             print("○ ", end="")
#     print()






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

# for i in range(len(lis) - 1):
#     print_board_npz(lis[i]- 40, lis[i] + 40, npz_dir, SIZE=19)



print_board_npz(4000, 5000, 10, npz_dir, SIZE=9)