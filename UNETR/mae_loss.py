import torch.nn as nn
from einops.layers.torch import Rearrange
  

class MaeLoss(nn.Module):
  def __init__(self, spatial_dims, feature_size):
    super().__init__()
    patch_size = [feature_size] * spatial_dims
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
    from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
    axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
    self.patches = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)

  def forward(self, src, _):
    pred, mask, inp = src
    loss = (self.patches(pred) - self.patches(inp)) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * mask).sum() / mask.sum()
    return loss