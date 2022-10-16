import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
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
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# modules used in FPS sampling
# farthest point sampling
def farthest_point_sampling(xyz, num_sumpling):
    """function to get indices of farthest point sampling

    Args:
        xyz (torch.tensor): (B, C, N)
        num_smpling (int): number of sampling

    Returns:
        torch.tensor(dtype=torch.long): (B, num_sumpling) This is indices of FPS sampling
    """

    device = xyz.device
    B, C, N = xyz.shape

    centroids = torch.zeros((B, num_sumpling), dtype=torch.long, device=device) # initialization of list for centroids
    farthest = torch.randint(0, N, size=(B,), device=device) # making initial centroids
    distance = torch.ones((B, N), dtype=torch.long, device=device)*1e10 # initialization of the nearest point lost

    batch_indicies = torch.arange(B, dtype=torch.long, device=device) # This is used to specify batch index.

    for i in range(num_sumpling):
        centroids[:, i] = farthest # updating list for centroids
        centroid = xyz[batch_indicies, :, farthest] # centriud has points cordinate of farthest
        centroid = centroid.view(B, C, 1) # reshape for compute distance between centroid and points in xyz
        dist = torch.sum((centroid - xyz)**2, dim=1) # computing distance
        mask = dist < distance # make boolean list
        distance[mask] = dist[mask] # update nearest list
        farthest = torch.max(distance, dim=1)[1] # update farthest ([1] means indices)
    
    return centroids

# changes indices to cordinates of points
def index2point_converter(xyz, indices):
    """converter that convert indices to cordinate

    Args:
        xyz (tensor): original points clouds (B, C, N)
        indices (tensor): indices that represent sampling points (B, num_sampling)

    Returns:
        tensor: (B, C, num_sampling)
    """
    device = xyz.device
    B, C, N = xyz.shape
    num_new_points = indices.shape[1]

    batch_indices = torch.arange(B, device=device)
    batch_indices = batch_indices.view([B, 1])
    batch_indices = batch_indices.repeat([1, num_new_points])

    new_xyz = xyz[batch_indices, :, indices]
    return new_xyz.permute(0, 2, 1)
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(10, 1026, 100)
    MLP = SharedMLP(1026, 1026)
    out = MLP(x)
    print(out.shape)
