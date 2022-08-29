import torch
import torch.nn as nn
from encoder import *
from decoder import *

class MSN(nn.Module):
    def __init__(self):
        super(MSN, self).__init__()
