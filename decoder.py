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
            nn.Conv1d(256, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.MLP_one(x)
        return out

# class for residual network
class PointNetResDecoder(nn.Module):
    def __init__(self, num_points):
        super(PointNetResDecoder, self).__init__()
        self.num_points = num_points

        self.MLP1 = SharedMLP(4, 64)
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 128),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        self.MaxPool = MaxPool(1024, self.num_points)
        self.MLP3 = nn.Sequential(
            SharedMLP(1088, 512),
            SharedMLP(512, 256),
            SharedMLP(256, 128),
            nn.Conv1d(128, 3, 1),
            nn.Tanh()
        )

    def forward(self, x):

        point_feature = self.MLP1(x)
        print(point_feature.shape)

        x = self.MLP2(point_feature)
        global_feature = self.MaxPool(x).view(-1, 1024, 1)
        global_feature = global_feature.repeat(1, 1, self.num_points)

        features = torch.cat([point_feature, global_feature], dim=1)

        out = self.MLP3(features)

        return out



if __name__ == "__main__":
    # x = torch.randn(10, 1026, 100)
    # mbd = OneMorphingDecoder(1026, "cpu")
    # out = mbd(x)
    # print(out.shape)

    x= torch.randn(10, 4, 100)
    pnrd = PointNetResDecoder(100)
    out = pnrd(x)
    print(out.shape)
