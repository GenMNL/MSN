import torch
import numpy as np
import open3d as o3d
import os
import sys
sys.path.append("./../emd")
import emd_module as emd

class DataNormalization():
    def __init__(self):
        pass

    def __call__(self, ary):
        max = ary.max(axis=0)
        min = ary.min(axis=0)
        x_max, y_max, z_max = max[0], max[1], max[2]
        x_min, y_min, z_min = min[0], min[1], min[2]

        ary[:,0] -= x_min
        ary[:,1] -= y_min
        ary[:,2] -= z_min
        ary[:,0] /= (x_max - x_min)
        ary[:,1] /= (y_max - y_min)
        ary[:,2] /= (z_max - z_min)

        return ary, max, min

if __name__ == "__main__":
    DN = DataNormalization()

    comp_pc = o3d.io.read_point_cloud("./comp/1.ply")
    comp_pc = np.asarray(comp_pc.points)
    comp_pc = DN(comp_pc)[0]
    comp_pc = torch.tensor(comp_pc, dtype=torch.float, device="cuda")
    comp_pc = comp_pc.unsqueeze(dim=0)

    PCN_pc = o3d.io.read_point_cloud("./PCN/1.ply")
    PCN_pc = np.asarray(PCN_pc.points)
    PCN_pc = DN(PCN_pc)[0]
    PCN_pc = torch.tensor(PCN_pc, dtype=torch.float, device="cuda")
    PCN_pc = PCN_pc.unsqueeze(dim=0)

    MSN_pc = o3d.io.read_point_cloud("./MSN/1.ply")
    MSN_pc = np.asarray(MSN_pc.points)
    MSN_pc = DN(MSN_pc)[0]
    MSN_pc = torch.tensor(MSN_pc, dtype=torch.float, device="cuda")
    MSN_pc = MSN_pc.unsqueeze(dim=0)

    PFNet_pc = o3d.io.read_point_cloud("./PF-Net/1.ply")
    PFNet_pc = np.asarray(PFNet_pc.points)
    PFNet_pc = DN(PFNet_pc)[0]
    PFNet_pc = torch.tensor(PFNet_pc, dtype=torch.float, device="cuda")
    PFNet_pc = PFNet_pc.unsqueeze(dim=0)

    emd_distance = emd.emdModule()
    eps = 0.002
    iters = 10000

    PCN_dist, _ = emd_distance(PCN_pc, comp_pc, eps, iters)
    MSN_dist, _ = emd_distance(MSN_pc, comp_pc, eps, iters)
    PFNet_dist, _ = emd_distance(PFNet_pc, comp_pc, eps, iters)

    print(np.sqrt(PCN_dist.cpu()).mean())
    print(np.sqrt(MSN_dist.cpu()).mean())
    print(np.sqrt(PFNet_dist.cpu()).mean())

