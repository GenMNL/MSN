import torch
import numpy as np

tens = torch.randn((2, 3, 5))
idx = np.zeros((2, 5))
for b in range(2):
    id = np.arange(5, dtype=int)
    id = np.random.permutation(id)
    idx[b, :] = id
idx = torch.tensor(idx, dtype=int)

batch_indices = torch.arange(2, dtype=torch.long)
batch_indices = batch_indices.view(2, 1)
batch_indices = batch_indices.repeat(1, 5)

print(idx)
print(tens)
print(tens[batch_indices, :, idx].permute(0, 2, 1))
# print(tens[idx])

# idx = np.repeat(idx, )
# print(idx)
