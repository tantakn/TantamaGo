import torch
from torch2trt import torch2trt
# from torchvision import models


import torch
import numpy as np
import psutil
import datetime

import sys
sys.path.append('../')
from nn.network import DualNet 



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


# # create some regular pytorch model...
# weights = models.ResNet50_Weights.IMAGENET1K_V1

# model = models.resnet50(weights=weights).eval().cuda()

model = DualNet(torch.device("cuda"))

# create example data
x = tmp_load_data_set("../backup/data_Q50000/sl_data_0.npz")
x = x.cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])