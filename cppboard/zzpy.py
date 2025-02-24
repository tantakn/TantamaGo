import onnx
import onnxruntime
import numpy as np
from onnxruntime.datasets import get_example
from onnx import mapping

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json

secret = json.load(open("/home/tantakn/code/TantamaGo/cppboard/gitignor_it.json"))

print(secret["ip_desk_ubuntu"])

import socket
# ソケットを作成
# client_socketというソケットオブジェクトを作成しています。
# socket.AF_INETはIPv4アドレスファミリを指定します。
# socket.SOCK_STREAMはTCPプロトコル（ストリームベースの通信）を指定します。
# これにより、IPv4のTCPソケットが生成されます。
client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# サーバーに接続
# 接続先は'localhost'（自分自身のマシン）で、ポート番号は8000です。
# サーバー側でserver_socket.accept()が実行され、接続待ちの状態である必要があります。
client_socket1.connect((secret["ip_desk_ubuntu"], int(secret["port"])))
client_socket2.connect((secret["ip_desk_ubuntu"], secret["port2"]))



# # onnx_model_path = "./test9_2.onnx"
# onnx_model_path = "./test19_2.onnx"
# # onnx_model_path = "./test2.onnx"
# # onnx_model_path = "TantamaGo/cppboard/test4.onnx"

# npz_path = "../backup/data_Q50000/sl_data_0.npz"
# # npz_path = "../backup/kgs-19-2019-04/sl_data_0.npz"

# print("onnx_model_path:", onnx_model_path)
# model = onnx.load(onnx_model_path)
# onnx.checker.check_model(model)

# # # モデルのグラフの表示
# # print(onnx.helper.printable_graph(model.graph))

# # インプットの形式の表示
# print("Inputs:")
# for input in model.graph.input:
#     tensor_type = input.type.tensor_type
#     elem_type = tensor_type.elem_type
#     type_str = mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, "Unknown")
#     print(f"{input.name} - {tensor_type.shape} - {type_str}")

# # アウトプットの形式の表示
# print("Outputs:")
# for output in model.graph.output:
#     tensor_type = output.type.tensor_type
#     elem_type = tensor_type.elem_type
#     type_str = mapping.TENSOR_TYPE_TO_NP_TYPE.get(elem_type, "Unknown")
#     print(f"{output.name} - {tensor_type.shape} - {type_str}")


# # -----推論の実行-----
# import torch

# def tmp_load_data_set(npz_path):


#     data = np.load(npz_path)


#     plane_data = data["input"].astype(np.float32)
#     policy_data = data["policy"].astype(np.float32)
#     value_data = data["value"].astype(np.int64)

#     plane_data = torch.tensor(plane_data)
#     policy_data = torch.tensor(policy_data)
#     value_data = torch.tensor(value_data)

#     return plane_data


# # 推論セッションの作成（CUDAを使用できない場合はCPUのみを使用）
# sess = onnxruntime.InferenceSession(onnx_model_path, 
#                                     providers=['CPUExecutionProvider'])
# # sess = onnxruntime.InferenceSession("test19.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# # インプット名と形状の取得
# input_name = sess.get_inputs()[0].name
# input_shape = sess.get_inputs()[0].shape

# # シンボリックな次元（例: 'batch_size'）を整数に置き換える
# fixed_shape = []
# for dim in input_shape:
#     if isinstance(dim, str):
#         fixed_shape.append(1)  # シンボリックな次元を1に設定
#     else:
#         fixed_shape.append(dim)
# fixed_shape = tuple(fixed_shape)

# input_data = tmp_load_data_set(npz_path)
# input_data = input_data.float()
# input_data = input_data[0].unsqueeze(0).to('cpu').numpy()
# # # ダミー入力データの作成（ランダムなデータを使用）
# # input_data = np.random.randn(*fixed_shape).astype(np.float32)

# # 推論の実行
# outputs = sess.run(None, {input_name: input_data})

# # 出力の表示
# print("推論結果:")
# for output in outputs:
#     print("output.shape:", output.shape)
#     print(output)