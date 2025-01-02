import torch
import numpy as np
import psutil
import datetime
import os

import sys
sys.path.append('../')
from nn.network import DualNet 

BATCH_SIZE = 1

with torch.no_grad():
    # デバイスを CUDA（GPU）に設定
    device = torch.device("cuda")
    # モデルをロードし、GPU に移動
    network = DualNet(device)
    network.to(device)
    network.load_state_dict(torch.load("/home/tantakn/code/TantamaGo/model/sl-model_20241020_214243_Ep:14.bin"))
    # network.load_state_dict(torch.load("/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin"))
    network.eval()
    torch.set_grad_enabled(False)

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

plane = tmp_load_data_set("../backup/data_Q50000/sl_data_0.npz")
# plane = tmp_load_data_set("../data/sl_data_0.npz")

model = network.float()
# dummy_input = plane
dummy_input = plane.float()

for name, param in model.named_parameters():
    print(f"{name}: {param.dtype}")

print(model)
print(dummy_input.shape)
print(dummy_input.dtype)
print(dummy_input[0])

torch.onnx.export(model,
                  dummy_input[0].unsqueeze(0).to(device),  # バッチ次元を追加
                  "./test3.onnx", export_params=True,
                  opset_version=10,
                  verbose=True, do_constant_folding=True, input_names=["input"],
                  output_names=["output_policy", "output_value"],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output_policy': {0: 'batch_size'},
                                'output_value': {0: 'batch_size'}}
                )
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