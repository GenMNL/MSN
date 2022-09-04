import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from module import *

class STNkd(nn.Module):
    def __init__(self, num_channels, device):
        super(STNkd, self).__init__()
        self.num_channels = num_channels
        self.device = device

        self.Conv_ReLU = nn.Sequential(
            SharedMLP(self.num_channels, 64),
            SharedMLP(64, 128),
            SharedMLP(128, 1024)
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.num_channels**2)
        )

    def forward(self, input):
        batchsize = input.shape[0]

        x = self.Conv_ReLU(input)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)

        iden = np.eye(self.num_channels).flatten().astype(np.float32)
        iden = Variable(torch.from_numpy(iden)).view(1, self.num_channels**2).repeat(batchsize, 1)
        iden = iden.to(self.device)

        out = x + iden
        out = out.view(-1, self.num_channels, self.num_channels)

        return out

if __name__ == "__main__":
    input = torch.randn(10, 3, 10, device="cuda")
    stn = STNkd(3, "cuda").to("cuda")
    out = stn(input)
    print(out.shape)
