import torch
import torch.nn as nn
from encoder import *
from decoder import *
import sys
sys.path.append("./expansion_penalty")
sys.path.append("./MDS")
import expansion_penalty_module as expansion
import MDS_module

class MSN(nn.Module):
    def __init__(self, emb_dim, num_output_points, num_surfaces, device):
        super(MSN, self).__init__()
        self.emb_dim = emb_dim
        self.num_output_points = num_output_points
        self.num_surfaces = num_surfaces
        self.device = device

        self.encoder = Encoder(self.emb_dim, self.device)
        self.coarse_decoder = nn.ModuleList([OneMorphingDecoder(self.emb_dim+2) for i in range(0, self.num_surfaces)])
        self.residual_decoder = PointNetResDecoder()

        self.expansion = expansion.expansionPenaltyModule()

    def forward(self, x):
        batchsize, _, _ = x.shape
        partial = x
        features = self.encoder(x)

        coarse_output = []
        for k in range(0, self.num_surfaces):
            rand_grid = torch.rand((batchsize, 2, self.num_output_points//self.num_surfaces), 
                                    dtype=torch.float32, 
                                    device=self.device)
            x = features.unsqueeze(dim=2).repeat(1, 1, rand_grid.shape[2])
            x = torch.cat([rand_grid, x], dim=1)
            one_coarse = self.coarse_decoder[k](x)
            coarse_output.append(one_coarse)

        # coarse output (tensor [b, c, n])
        coarse_output = torch.cat(coarse_output, dim=2)
        coarse_output = coarse_output.transpose(1, 2).contiguous()

        # get expansion loss
        # mean_mst_dis is the means of points distance. It is used for var of MDS
        dist, _, mean_mst_dis = self.expansion(coarse_output, self.num_output_points//self.num_surfaces, 1.5)
        loss_mst = torch.mean(dist)

        coarse_output = coarse_output.transpose(1, 2).contiguous()

        # get id of input partial points and coarse output poitns
        id_partial = torch.zeros(batchsize, 1, partial.shape[2], device=self.device)
        x_partial = torch.cat([partial, id_partial], dim=1)
        id_coarse = torch.ones(batchsize, 1, coarse_output.shape[2], device=self.device)
        x_coarse = torch.cat([coarse_output, id_coarse], dim=1)
        # concatnate partial input points and coarse output points which have identifier index
        x = torch.cat([x_partial, x_coarse], dim=2) # [batch, 4(xyz+identifier), num_points(partial+coarse)]

        # get index of minimun density sampling
        # the sampling num is equall with num_coarse_points
        MDS_index = MDS_module.minimum_density_sample(x[:, 0:3, :].transpose(1, 2).contiguous(), x_coarse.shape[2], mean_mst_dis) 
        x = MDS_module.gather_operation(x, MDS_index) # [batch, 4(xyz+identifier), num_points(coarse)]

        # This is decoder to get fine output
        # the num points of fine and coarse is same, but the accuracy of fine is more than coarse
        # fine residual moves each points of coarse output to correct direction.
        fine_residual = self.residual_decoder(x)
        fine_output = x[:, 0:3, :] + fine_residual

        return coarse_output, fine_output, loss_mst

if __name__ == "__main__":
    input = torch.randn(10, 3, 1024, device="cuda")

    model = MSN(1024, 1024, 16, "cuda").to("cuda")

    coarse_output, fine_output, loss= model(input)
    print(coarse_output.shape)
    print(fine_output.shape)

