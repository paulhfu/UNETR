import torch

def mae_loss(src, _):
  pred, mask, patches = src
  loss = (pred - patches) ** 2
  loss = loss.mean(dim=-1)
  loss = (loss * mask).sum() / mask.sum()
  return loss