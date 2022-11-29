import torch
from UNETR.model import UNETR
from UnetPretrained.unet import UNet2d
import torch_em
from torch_em.data.datasets import get_livecell_loader
#https://bioimage.io/#/?tags=Livecell&id=10.5281%2Fzenodo.5869899

def eval2DUnet():
    patch_shape = [512, 512]
    batch_size = 10
    test_loader = get_livecell_loader(
        "/home/drford/data", patch_shape, "test",
        boundaries=True, batch_size=batch_size
    )
    
    model = UNet2d(
      depth=4,
      final_activation= "Sigmoid",
      gain=2,
      in_channels=1,
      initial_features=64,
      out_channels=2,
      postprocessing=None,
      return_side_outputs=False
    )
    model.load_state_dict(torch.load("UnetPretrained/weights.pt"))
    loss = torch_em.loss.DiceLoss()
    cumLoss = 0.0
    for src, tgt in test_loader:
        cumLoss += loss(model(src), tgt)
    cumLoss /= len(test_loader)
    print(f"Cum test score: {1 - cumLoss}")

def eval2DUnetR():
    patch_shape = [512, 512]
    batch_size = 10
    test_loader = get_livecell_loader(
        "/home/drford/data", patch_shape, "test",
        boundaries=True, batch_size=batch_size
    )
    
    model = UNETR(
          in_channels=1,
          out_channels=2,
          img_size=patch_shape,
          feature_size = 16,
          hidden_size = 768,
          mlp_dim = 3072,
          num_heads = 12,
          conv_block = True,
          dropout_rate = 0.0,
          masked_pretrain = False)
    
    model.load_state_dict(torch.load("bestCkPnt.pt"))
    loss = torch_em.loss.DiceLoss()
    cumLoss = 0.0
    for src, tgt in test_loader:
        cumLoss += loss(model(src), tgt)
    cumLoss /= len(test_loader)
    print(f"Cum test score: {1 - cumLoss}")

if __name__ == '__main__':
    eval2DUnet()
    eval2DUnetR()