import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
  

class MaeLoss(nn.Module):
  def __init__(self, spatial_dims, feature_size, n_patches):
    super().__init__()
    patch_size = [feature_size] * spatial_dims
    chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
    from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
    to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
    axes_len = {f"p{i+1}": p for i, p in enumerate(patch_size)}
    self.patches = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)
    self.patchesInverse = Rearrange(f"{to_chars} -> {from_chars}", h=n_patches[0], w=n_patches[1], **axes_len)

  def forward(self, src, _):
    loss = (self.patches(src[0]) - self.patches(src[2])) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss * src[1]).sum() / src[1].sum()
    spatial_mask = self.patchesInverse(self.patches(torch.ones_like(src[0])) * src[1][..., None])
    src.append(spatial_mask)
    return loss