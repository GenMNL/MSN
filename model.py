import torch
import torch.nn as nn
from encoder import *
from decoder import *

class MSN(nn.Module):
    def __init__(self, num_points, emb_dim, num_surfaces, device):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.device = device
        self.num_surfaces = num_surfaces

        self.encoder = Encoder(self.num_points, self.emb_dim, self.device)

    def forward(self, x):
        batchsize, _, _ = x.shape
        partial = x
        features = self.encoder(x)

        out = []
        for k in range(0, self.num_surfaces):
            rand_grid = torch.rand((batchsize, 2, self.num_points//self.num_points), device=self.device)
