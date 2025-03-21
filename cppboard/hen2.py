import torch
import numpy as np
import psutil
import datetime
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import sys
sys.path.append('../')
from nn.network import DualNet
from nn.network import DualNet_256_24
from torch.nn.parallel import DataParallel
from torch.serialization import add_safe_globals



save_onnx_path = "./9_rl-model_default.onnx"

BOARD_SIZE = 9
dummy_npz_path = "../backup/data_Q50000/sl_data_0.npz"
# model_path = "/home/tantakn/code/TantamaGo/model_def/sl-model_q50k_DualNet_256_24.bin"
model_path = "../model_def/rl-model_default.bin"
# model_path = "/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin"
# model_path = "/home/tantakn/code/TantamaGo/model/sl-model_20250304_005752_240.bin"

# BOARD_SIZE = 13
# dummy_npz_path = "/home/tantakn/code/TantamaGo/backup/13_2_0.npz"
# # model_path = "/home/tantakn/code/TantamaGo/model_def/sl-model_20250227_033544_Ep00_13_1.bin"

# BOARD_SIZE = 19
# dummy_npz_path = "../backup/kgs-19-2019-04/sl_data_0.npz"
# model_path = "/home/tantakn/code/TantamaGo/model_def/sl-model_20250110_031407_19.bin"



# model_path = "/home/tantakn/code/TantamaGo/model/sl-model_20250303_225555_370.bin"
# model_path = "/home/tantakn/code/TantamaGo/model/sl-model_20250125_025418.bin"
# model_path = "/home/tantakn/code/TantamaGo/model/sl-model_20241020_214243_Ep:14.bin"



BATCH_SIZE = 1



# DataParallelクラスを安全なクラスとして明示的に登録
add_safe_globals([DataParallel])

with torch.no_grad():
    # デバイスを CUDA（GPU）に設定
    device = torch.device("cuda")
    
    # モデルを一度だけインスタンス化する
    network = DualNet(device, BOARD_SIZE)
    # network = DualNet_256_24(device, BOARD_SIZE)
    network.to(device)

    # DataParallelで保存されたモデルを適切にロードする
    try:
        # weights_only=Trueを使用して、DataParallelを許可リストに追加した上で安全にロード
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 以下は既存のコードと同じ
        if isinstance(state_dict, torch.nn.parallel.DataParallel):
            state_dict = state_dict.module.state_dict()
        elif not isinstance(state_dict, dict):
            print(f"警告: 予期しないモデル型 {type(state_dict)}")
            if hasattr(state_dict, 'state_dict'):
                state_dict = state_dict.state_dict()
        
        network.load_state_dict(state_dict)
        print("モデルを正常にロードしました")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        # 代替方法: セキュリティリスクを受け入れる場合はweights_only=Falseを試す
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            if isinstance(state_dict, torch.nn.parallel.DataParallel):
                state_dict = state_dict.module.state_dict()
            elif not isinstance(state_dict, dict):
                if hasattr(state_dict, 'state_dict'):
                    state_dict = state_dict.state_dict()
            network.load_state_dict(state_dict)
            print("代替方法でモデルを正常にロードしました")
        except Exception as e2:
            print(f"代替方法でもモデルのロード中にエラーが発生しました: {e2}")
            raise


# with torch.no_grad():
#     # デバイスを CUDA（GPU）に設定
#     device = torch.device("cuda")
#     # モデルをロードし、GPU に移動
#     network = DualNet_256_24(device, BOARD_SIZE)
#     # network = DualNet(device, BOARD_SIZE)
#     network.to(device)

#     network.load_state_dict(torch.load(model_path))
#     # DataParallelで保存されたモデルのstate_dictを修正
#     state_dict = torch.load(model_path)
#     if isinstance(state_dict, torch.nn.parallel.DataParallel):
#         state_dict = state_dict.module.state_dict()

#     network = DualNet_256_24(device, BOARD_SIZE)
#     # network = DualNet(device, BOARD_SIZE)
#     network.to(device)

#     # DataParallelで保存されたモデルのstate_dictを修正
#     state_dict = torch.load(model_path, map_location='cpu')  # まずCPUにロード
#     if isinstance(state_dict, torch.nn.parallel.DataParallel):
#         state_dict = state_dict.module.state_dict()
#     network.load_state_dict(state_dict)
#     # # DataParallelで保存されたモデルのstate_dictを修正
#     # from collections import OrderedDict
#     # new_state_dict = OrderedDict()
#     # for k, v in network.items():
#     #     if k.startswith('module.'):
#     #         name = k[7:] # module.を取り除く
#     #         new_state_dict[name] = v
#     #     else:
#     #         new_state_dict[k] = v
        
#     # network.load_state_dict(new_state_dict)
#     # network.load_state_dict(torch.load(model_path))

#     network.eval()
#     torch.set_grad_enabled(False)

#     # 入力データを GPU に移動
#     x = torch.ones((BATCH_SIZE, 6, 9, 9)).to(device)
    
    
# device = torch.device("cuda")
# model = DualNet(device)
# model.load_state_dict(torch.load("/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin"))
# model.to(device)  # モデルをGPUに移動
# model.eval()

def tmp_load_data_set(npz_path):
    def check_memory_usage():
        if not psutil.virtual_memory().percent < 90:
            print(f"memory usage is too high. mem_use: {psutil.virtual_memory().percent}% [{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}]")
            assert True

    check_memory_usage()

    data = np.load(npz_path)

    check_memory_usage()

    plane_data = data["input"].astype(np.float32)
    policy_data = data["policy"].astype(np.float32)
    value_data = data["value"].astype(np.int64)

    check_memory_usage()

    plane_data = torch.tensor(plane_data)
    policy_data = torch.tensor(policy_data)
    value_data = torch.tensor(value_data)

    return plane_data


plane = tmp_load_data_set(dummy_npz_path)

model = network.float()
# dummy_input = plane
dummy_input = plane.float()

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

print("model: ", model)
print("dummy_input.shape: ", dummy_input.shape)
print("dummy_input.dtype: ", dummy_input.dtype)
print("dummy_input[0]: ", dummy_input[0])


# ONNXエクスポート（fixed batch size）
torch.onnx.export(model,
                  dummy_input[0].unsqueeze(0).to(device),  # バッチ次元を追加
                  save_onnx_path, 
                  export_params=True,
                  opset_version=10,
                  verbose=True, 
                  do_constant_folding=True, 
                  input_names=["input"],
                  output_names=["output_policy", "output_value"])
                  # dynamic_axes パラメータを削除
""" 出力例
onnx_model_path: ./test9_2.onnx
Inputs:
input - dim {
  dim_value: 1
}
dim {
  dim_value: 6
}
dim {
  dim_value: 9
}
dim {
  dim_value: 9
}
 - float32
Outputs:
output_policy - dim {
  dim_value: 1
}
dim {
  dim_value: 82
}
 - float32
output_value - dim {
  dim_value: 1
}
dim {
  dim_value: 3
}
 - float32
"""




# torch.onnx.export(model,
#                   dummy_input[0].unsqueeze(0).to(device),  # バッチ次元を追加
#                   save_onnx_path, export_params=True,
#                   opset_version=10,
#                   verbose=True, do_constant_folding=True, input_names=["input"],
#                   output_names=["output_policy", "output_value"],
#                   dynamic_axes={'input': {0: 'batch_size'},
#                                 'output_policy': {0: 'batch_size'},
#                                 'output_value': {0: 'batch_size'}}
#                 )
""" 出力例 バッチ次元が可変？
(envGo) tantakn@DESKTOP-C96CIQ7:~/code/TantamaGo/cppboard$ python3 ./zzpy.py 
onnx_model_path: ./test9_1.onnx
Inputs:
input - dim {
  dim_param: "batch_size"
}
dim {
  dim_value: 6
}
dim {
  dim_value: 9
}
dim {
  dim_value: 9
}
 - float32
Outputs:
output_policy - dim {
  dim_param: "batch_size"
}
dim {
  dim_value: 82
}
 - float32
output_value - dim {
  dim_param: "batch_size"
}
dim {
  dim_value: 3
}
 - float32"""






# torch.onnx.export(model,
#                   dummy_input[0].unsqueeze(0).to(device),  # バッチ次元を追加
#                   "./test.onnx", export_params=True,
#                   opset_version=8,
#                   verbose=True, do_constant_folding=True, input_names=["input"],
#                   output_names=["output_policy", "output_value"],
#                   dynamic_axes={'input': {0: 'batch_size'},
#                                 'output_policy': {0: 'batch_size'},
#                                 'output_value': {0: 'batch_size'}})
# torch.onnx.export(model,
#                   plane[0].unsqueeze(0).to(device),  # バッチ次元を追加
#                   "./test.onnx", export_params=True,
#                   opset_version=10,
#                   verbose=True, do_constant_folding=True, input_names=["input"],
#                   output_names=["output_policy", "output_value"],
#                   dynamic_axes={'input': {0: 'batch_size'},
#                                 'output_policy': {0: 'batch_size'},
#                                 'output_value': {0: 'batch_size'}})


# torch.onnx.export(model, dummy_input, "test.onnx", opset_version=9)


# Unsupported ONNX data type: DOUBLE (11)
# Parameter check failed at: ../builder/Layers.cpp::ConstantLayer::2278, condition: weights.type == DataType::kFLOAT || weights.type == DataType::kHALF || weights.type == DataType::kINT32
# While parsing node number 36 [Cast]:
# ERROR: builtin_op_importers.cpp:727 In function importCast:
# [8] Assertion failed: trt_dtype == nvinfer1::DataType::kHALF && cast_dtype == ::ONNX_NAMESPACE::TensorProto::FLOAT
# Network must have at least one output
# Segmentation fault (コアダンプ)Unsupported ONNX data type: DOUBLE (11)


# Unsupported ONNX data type: DOUBLE (11)
# Parameter check failed at: ../builder/Layers.cpp::ConstantLayer::2278, condition: weights.type == DataType::kFLOAT || weights.type == DataType::kHALF || weights.type == DataType::kINT32
# While parsing node number 36 [Cast]:
# ERROR: builtin_op_importers.cpp:727 In function importCast:
# [8] Assertion failed: trt_dtype == nvinfer1::DataType::kHALF && cast_dtype == ::ONNX_NAMESPACE::TensorProto::FLOAT
# Network must have at least one output
# Segmentation fault (コアダンプ)

# Unsupported ONNX data type: DOUBLE (11)
# Parameter check failed at: ../builder/Layers.cpp::ConstantLayer::2278, condition: weights.type == DataType::kFLOAT || weights.type == DataType::kHALF || weights.type == DataType::kINT32
# While parsing node number 36 [Cast]:
# ERROR: builtin_op_importers.cpp:727 In function importCast:
# [8] Assertion failed: trt_dtype == nvinfer1::DataType::kHALF && cast_dtype == ::ONNX_NAMESPACE::TensorProto::FLOAT
# Network must have at least one output
# Segmentation fault (コアダンプ)