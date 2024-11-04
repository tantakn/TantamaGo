import torch
import numpy as np
import psutil
import datetime

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
    network.load_state_dict(torch.load("/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin"))
    network.eval()
    torch.set_grad_enabled(False)

    # 入力データを GPU に移動
    x = torch.ones((BATCH_SIZE, 6, 9, 9)).to(device)
    
    
device = torch.device("cuda")
model = DualNet(device)
model.load_state_dict(torch.load("/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin"))
model.to(device)  # モデルをGPUに移動
model.eval()

def tmp_load_data_set(npz_path):
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

    return plane_data



plane = tmp_load_data_set("../data/sl_data_0.npz")

print(plane.shape)
print(plane[0])

torch.onnx.export(model,
                  plane[0].unsqueeze(0).to(device),  # バッチ次元を追加
                  "./test.onnx", export_params=True,
                  opset_version=10,
                  verbose=True, do_constant_folding=True, input_names=["input"],
                  output_names=["output_policy", "output_value"],
                  dynamic_axes={'input': {0: 'batch_size'},
                                'output_policy': {0: 'batch_size'},
                                'output_value': {0: 'batch_size'}})