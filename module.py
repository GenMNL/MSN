import torch
import torch.nn as nn

class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out

class MaxPool(nn.Module):
    def __init__(self, num_channels, num_points):
        super(MaxPool, self).__init__()
        self.num_channels = num_channels
        self.num_points = num_points
        self.main = nn.MaxPool1d(self.num_points)

    def forward(self, x):
        x = x.view(-1, self.num_channels, self.num_points)
        out = self.main(x)
        out = out.view(-1, self.num_channels)
        return out

if __name__ == "__main__":
    x = torch.randn(10, 1026, 100)
    MLP = SharedMLP(1026, 1026)
    mp = MaxPool(1026, 100)
    out = MLP(x)
    out = mp(out)
    print(out.shape)
