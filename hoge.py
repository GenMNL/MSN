import torch

grid = torch.FloatTensor(1, 3)
print(grid.data.uniform_(0, 1))
