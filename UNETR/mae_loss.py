import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
  

class MaeLoss(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, src, _):
    logits, mask, input_data, patch = src
    loss = (patch(logits) - patch(input_data)) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss