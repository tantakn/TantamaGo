"""Policy headの実装。
"""
import torch
from torch import nn


class PolicyHead(nn.Module):
    """Policy headの実装クラス。
    """
    def __init__(self, board_size: int, channels: int, momentum: float=0.01):
        """Policy headの初期化処理。

        Args:
            board_size (int): 碁盤のサイズ。
            channels (int): 共通ブロック部の畳み込み層のチャネル数。
            momentum (float, optional): バッチ正則化層のモーメンタムパラメータ. Defaults to 0.01.
        """
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=channels, out_channels=2, \
            kernel_size=1, padding=0, bias=False)
        self.bn_layer = nn.BatchNorm2d(num_features=2, eps=2e-5, momentum=momentum)
        self.fc_layer = nn.Linear(2 * board_size ** 2, board_size ** 2 + 1)
        self.relu = nn.ReLU()

    def forward(self, input_plane: torch.Tensor) -> torch.Tensor:
        """前向き伝搬処理を実行する。

        Args:
            input_plane (torch.Tensor): Policy headへの入力テンソル。

        Returns:
            torch.Tensor: PolicyのLogit出力
        """
        # hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        # batch_size = hidden.size(0)

        # # total_size の計算を省略し、-1 を使用して自動的にサイズを推定
        # reshape = hidden.view(batch_size, -1)

        # policy_out = self.fc_layer(reshape)


        #---------------
        # batch_size, channels, height, width = hidden.shape

        # # height と width を明示的に int 型に変換
        # height = int(height)
        # width = int(width)
        # channels = int(channels)
        # batch_size = int(batch_size)

        # total_size = channels * height * width
        # total_size = int(total_size)
        # reshape = hidden.reshape(batch_size, total_size)

        # policy_out = self.fc_layer(reshape)


        #---------------
        hidden = self.relu(self.bn_layer(self.conv_layer(input_plane)))
        batch_size, channels, height, witdh = hidden.shape
        reshape = hidden.reshape(batch_size, channels * height * witdh)

        policy_out = self.fc_layer(reshape)



        return policy_out
