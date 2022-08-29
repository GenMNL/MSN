import torch
import torch.nn as nn
from module import *
from stn import *

class Encoder(nn.Module):
    def __init__(self, num_points, emb_dim, device):
        super(Encoder, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.device = device

        self.stn3d = STNkd(3, self.num_points, self.device)
        self.MLP1 = nn.Sequential(
            SharedMLP(3, 64),
            SharedMLP(64, 64)
        )
        self.stn64d = STNkd(64, self.num_points, self.device)
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
        )
        self.MaxPool = MaxPool(1024, self.num_points)
        self.fc = nn.Sequential(
            nn.Linear(1024, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # stn for input
        trans_3d = self.stn3d(x)
        x = x.permute(0, 2, 1)
        trans_x = torch.bmm(x, trans_3d)
        trans_x = trans_x.permute(0, 2, 1)

        # MLP1
        x = self.MLP1(trans_x)

        # stn for second t-net
        trans_64d = self.stn64d(x)
        x = x.permute(0, 2, 1)
        trans_x = torch.bmm(x, trans_64d)
        trans_x = trans_x.permute(0, 2, 1)

        # MLP2
        x = self.MLP2(trans_x)

        # Max Pooling
        x = self.MaxPool(x)
        x = x.view(-1, 1024)

        # fully convolution
        out = self.fc(x)
        return out


if __name__ == "__main__":
    x = torch.randn(10, 3, 100)
    encoder = Encoder(100, 1024, "cpu")
    out = encoder(x)
    print(out.shape)
    print(out)
