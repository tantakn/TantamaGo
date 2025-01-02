"""Dual Networkの実装。
"""
from typing import Tuple
from torch import nn
import torch

from board.constant import BOARD_SIZE
from nn.network.res_block import ResidualBlock
from nn.network.head.policy_head import PolicyHead
from nn.network.head.value_head import ValueHead


class DualNet(nn.Module): # pylint: disable=R0902
    """Dual Networkの実装クラス。
    """
    def __init__(self, device: torch.device, board_size: int=BOARD_SIZE):
        """Dual Networkの初期化処理

        Args:
            device (torch.device): 推論実行デバイス。探索での推論実行時にのみ使用し、学習中には使用しない。
            board_size (int, optional): 碁盤のサイズ。 デフォルト値はBOARD_SIZE。
        """
        super().__init__()
        filters = 64
        blocks = 6

        self.device = device

        # resnet前の畳み込み層
        self.conv_layer = nn.Conv2d(in_channels=6, out_channels=filters, kernel_size=3, padding=1, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=filters)
        self.relu = nn.ReLU()
        self.blocks = make_common_blocks(blocks, filters)
        self.policy_head = PolicyHead(board_size, filters)
        self.value_head = ValueHead(board_size, filters)

        self.softmax = nn.Softmax(dim=1)

        self.filter_num = filters#########
        self.block_num = blocks
        self.input_type = ""


    def forward(self, input_plane: torch.Tensor, pram: str="") -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: PolicyとValueのlogit。
        """

        if pram == "sl":
            return self.forward_for_sl(input_plane)

        # blocks がresnet
        blocks_out = self.blocks(self.relu(self.bn_layer(self.conv_layer(input_plane))))

        return self.policy_head(blocks_out), self.value_head(blocks_out)



    def forward_for_sl(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。教師有り学習で利用する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Softmaxを通したPolicyと, Valueのlogit
        """

        policy, value = self.forward(input_plane)
        return self.softmax(policy), value


    def forward_with_softmax(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane)
        return self.softmax(policy), self.softmax(value)


    def inference(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。探索用に使うメソッドのため、デバイス間データ転送も内部処理する。（たぶん、cpuに転送するという意味）

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane.to(self.device))
        return self.softmax(policy).cpu(), self.softmax(value).cpu()


    def inference_with_policy_logits(self, input_plane: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向き伝搬処理を実行する。Gumbel AlphaZero用の探索に使うメソッドのため、デバイス間データ転送も内部処理する。（たぶん、cpuに転送するという意味）
        inference() との違いは policy が softmax を通らないこと。

        Args:
            input_plane (torch.Tensor): 入力特徴テンソル。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Policy, Valueの推論結果。
        """
        policy, value = self.forward(input_plane.to(self.device))
        return policy.cpu(), self.softmax(value).cpu()


def make_common_blocks(num_blocks: int, num_filters: int) -> torch.nn.Sequential:
    """DualNetの共通の残差ブロック（多分、resnet）を構成して返す。

    Args:
        num_blocks (int): 積み上げる残差ブロック数。
        num_filters (int): 残差ブロック内の畳込み層のフィルタ数。

    Returns:
        torch.nn.Sequential: 残差ブロック列。
    """
    blocks = [ResidualBlock(num_filters) for _ in range(num_blocks)]
    return nn.Sequential(*blocks)





""" 簡単な実行例1
# 0: 空点, 1: 黒石, 2: 白石。最後の3つは、手番の色、手のy座標、手のx座標。
data = "[[3,3,3,3,3,3,3,3,3,3,3],[3,1,0,1,1,0,1,1,1,1,3],[3,0,1,1,0,1,1,1,2,0,3],[3,1,1,1,1,1,1,0,1,2,3],[3,1,1,1,1,1,1,1,2,2,3],[3,2,2,1,2,1,1,1,1,1,3],[3,2,2,2,2,1,0,1,1,1,3],[3,0,2,0,2,2,1,1,1,0,3],[3,2,2,2,0,2,1,1,1,1,3],[3,2,2,0,2,1,1,1,0,1,3],[3,3,3,3,3,3,3,3,3,3,3],[1,4,8]]"

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
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward(input_planes)
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward_for_sl(input_planes)
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.forward_with_softmax(input_planes)
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.inference(input_planes)
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)

policy_data, value_data = network.inference_with_policy_logits(input_planes)
print(policy_data, file=sys.stderr)
print(value_data, file=sys.stderr)
"""


""" 簡単な実行例2
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


tmp_npz = tmp_load_data_set("/home0/y2024/u2424004/igo/TantamaGo/backup/data_Q50000/sl_data_0.npz")

input_t = tmp_npz[0][1234].unsqueeze(0).to(device)  # バッチ次元を追加。1234番目の局面を取得
print(input_t.shape)######
print(input_t)######

policy_data, value_data = network.inference(input_t)

policy_data = policy_data.numpy()
print(np.sum(policy_data))######
policy_data = json.dumps(policy_data.tolist())
print(policy_data)######

value_data = value_data.numpy()
print(np.sum(value_data))######
value_data = json.dumps(value_data.tolist())
print(value_data)######
"""