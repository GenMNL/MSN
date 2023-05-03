import torch
import torch.nn as nn
from models.module import *
# from stn import *

class Encoder(nn.Module):
    def __init__(self, emb_dim):
        super(Encoder, self).__init__()
        self.emb_dim = emb_dim

        # self.stn3d = STNkd(3)
        self.MLP1 = nn.Sequential(
            SharedMLP(3, 64),
            SharedMLP(64, 64)
        )
        # self.stn64d = STNkd(64)
        self.MLP2 = nn.Sequential(
            SharedMLP(64, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, self.emb_dim),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # stn for input
        # trans_3d = self.stn3d(x)
        # x = x.permute(0, 2, 1)
        # trans_x = torch.bmm(x, trans_3d)
        # trans_x = trans_x.permute(0, 2, 1)

        # MLP1
        # x = self.MLP1(trans_x)
        x = self.MLP1(x)

        # stn for second t-net
        # trans_64d = self.stn64d(x)
        # x = x.permute(0, 2, 1)
        # trans_x = torch.bmm(x, trans_64d)
        # trans_x = trans_x.permute(0, 2, 1)

        # MLP2
        # x = self.MLP2(trans_x)
        x = self.MLP2(x)

        # Max Pooling
        x, _ = torch.max(x, dim=2)

        # fully convolution
        out = self.fc(x)
        return out


if __name__ == "__main__":
    x = torch.randn(10, 3, 100, device="cuda")
    encoder = Encoder(1024, "cuda").to("cuda")
    out = encoder(x)
    print(out.shape)
    print(out)
