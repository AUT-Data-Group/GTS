import math
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from logging import config
from einops import rearrange, reduce, repeat

try:
    from .model_ssl import ViViTSSL
except ImportError:
    from model_ssl import ViViTSSL


class FineTune(nn.Module):


    def __init__(self, checkpoint, **encoder_args):
        super().__init__()
        self.encoder = ViViTSSL.load_state_dict(checkpoint, **encoder_args)
        self.fc = nn.Linear(768, 414)
        self.conv = nn.Conv2d(1, 12, kernel_size=1)
    
    def forward(self, x):
        x = self.encoder(x)
        fc = self.fc(x[:, 0])
        return rearrange(self.conv(rearrange(fc.unsqueeze(1), "b x (n c) -> b x n c", n=207, c=2)), "b x n c -> x b (n c)")

