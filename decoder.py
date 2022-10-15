import torch
import torch.nn as nn
from module import *

# class for morphing based decoder
class OneMorphingDecoder(nn.Module):
    def __init__(self, in_channels):
        super(OneMorphingDecoder, self).__init__()
        self.in_channels = in_channels

        self.MLP_one = nn.Sequential(
            SharedMLP(self.in_channels, self.in_channels),
            SharedMLP(self.in_channels, 513),
            SharedMLP(513, 256),
            SharedMLP(256, 3)
            # nn.Conv1d(256, 3, 1),
            # nn.Tanh()
        )

    def forward(self, x):
        out = self.MLP_one(x)
        return out

# class for residual network
class PointNetResDecoder(nn.Module):
    def __init__(self):
        super(PointNetResDecoder, self).__init__()

        self.MLP1 = SharedMLP(4, 64)
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
            # nn.Conv1d(128, 1024, 1),
            # nn.BatchNorm1d(1024)
        )
        self.MLP3 = nn.Sequential(
            SharedMLP(1088, 512),
            SharedMLP(512, 256),
            SharedMLP(256, 128),
            SharedMLP(128, 3)
            # nn.Conv1d(128, 3, 1),
            # nn.Tanh()
        )

    def forward(self, x):
        """residual decoder

        Args:
            x (tensor): (B, 4, N)

        Returns:
            tensor: (B, 3, N)
        """

        point_feature = self.MLP1(x)

        x = self.MLP2(point_feature)
        global_feature, _ = torch.max(x, dim=2, keepdim=True)
        global_feature = global_feature.repeat(1, 1, point_feature.shape[2])

        features = torch.cat([point_feature, global_feature], dim=1)

        out = self.MLP3(features)

        return out



if __name__ == "__main__":
    # x = torch.randn(10, 1026, 100)
    # mbd = OneMorphingDecoder(1026, "cpu")
    # out = mbd(x)
    # print(out.shape)

    x= torch.randn(10, 4, 100)
    pnrd = PointNetResDecoder()
    out = pnrd(x)
    print(out.shape)
