import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmseg.registry import MODELS

@MODELS.register_module()
class ParseLoss(nn.Module):
    def __init__(self, num_losses=2) -> None:
        super().__init__()
        W = torch.ones((num_losses), dtype=torch.float32)
        self.W = nn.Parameter(W)
        self.deta = torch.tensor(1e-6)
    def forward(self, log_vars):
        t = 0
        loss = 0
        for key, value in log_vars:
            if 'loss' in key:
                loss += value/(2*(self.W[t] ** 2)+self.deta)
                t += 1
        for w in self.W:
            loss += torch.exp(w)
        return loss