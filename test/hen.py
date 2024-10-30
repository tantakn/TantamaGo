import torch
from torch2trt import torch2trt

print("ok")
# # create some regular pytorch model...
# weights = models.ResNet50_Weights.IMAGENET1K_V1

# model = models.resnet50(weights=weights).eval().cuda()

# # create example data
# x = torch.ones((1, 3, 224, 224)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])


from nn.network import DualNet # 自分のネットワークを定義したファイル

MODEL_NAME="MyModel"
TRT_NAME="MyTRTModel"

BATCH_SIZE=1


def load_network(model_file_path: str, use_gpu: bool, gpu_num: int=-1) -> DualNet:
    """ニューラルネットワークをロードして取得する。

    Args:
        model_file_path (str): ニューラルネットワークのパラメータファイルパス。
        use_gpu (bool): GPU使用フラグ。

    Returns:
        DualNet: パラメータロード済みのニューラルネットワーク。
    """
    device = torch.device("cuda")
    network = DualNet(device)
    network.to(device)
    try:
        network.load_state_dict(torch.load(model_file_path))
    except Exception as e: # pylint: disable=W0702
        print(f"Failed to load_network {model_file_path}.")
        raise("Failed to load_network.")
    network.eval()
    torch.set_grad_enabled(False)

    return network

with torch.no_grad():
    net=load_network("/home0/y2024/u2424004/igo/TantamaGo/model_def/sl-model_default.bin",True)
    net.cuda().eval().half() # fp16_modeの場合も、自分で.half()しておく必要はない（？）

    x=torch.rand((BATCH_SIZE,3,224,224)).cuda().half()

    model_trt=torch2trt(net,[x],fp16_mode=True,int8_mode=False,max_batch_size=BATCH_SIZE)

torch.save(model_trt.state_dict(),"trt_model/"+TRT_NAME+".pth")