import torch
import torch.nn as nn
from encoder import *
from decoder import *
import expansion_penalty as expansion

class MSN(nn.Module):
    def __init__(self, num_points, emb_dim, num_surfaces, device):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.emb_dim = emb_dim
        self.device = device
        self.num_surfaces = num_surfaces

        self.encoder = Encoder(self.num_points, self.emb_dim, self.device)
        self.coarse_decoder = nn.ModuleList([OneMorphingDecoder(self.emb_dim+2) for i in range(0, self.num_surfaces)])

    def forward(self, x):
        batchsize, _, _ = x.shape
        partial = x
        features = self.encoder(x)

        coarse_output = []
        for k in range(0, self.num_surfaces):
            rand_grid = torch.rand((batchsize, 2, self.num_points//self.num_points), 
                                    dtype=torch.float32, 
                                    device=self.device)
            x = features.unsqueeze(dim=2).repeat(1, 1, rand_grid.shape[2])
            x = torch.cat([rand_grid, x], dim=1)
            one_coarse = self.coarse_decoder[k](x)
            coarse_output.append(one_coarse)

        coarse_output = torch.cat(coarse_output, dim=2)

