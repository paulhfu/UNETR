import torch
from UNETR.model import UNETR
from UnetPretrained.unet import UNet2d
import torch_em
from torch_em.data.datasets import get_livecell_loader
from tqdm import tqdm
#https://bioimage.io/#/?tags=Livecell&id=10.5281%2Fzenodo.5869899

@torch.no_grad()
def eval2DUnet():
    patch_shape = [512, 512]
    batch_size = 10
    test_loader = get_livecell_loader(
        "/nfs/home/e7faffa3966db4c3/data", patch_shape, "test",
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
    model = model.cuda()
    model = model.eval()
    loss = torch_em.loss.DiceLoss()
    cumLoss = 0.0
    for src, tgt in tqdm(test_loader):
        src = src.to(torch.device("cuda"))
        tgt = tgt.to(torch.device("cuda"))
        cumLoss += loss(model(src), tgt)
    cumLoss /= len(test_loader)
    print(f"Unet Cum test score: {1 - cumLoss}")

@torch.no_grad()
def eval2DUnetR():
    patch_shape = [512, 512]
    batch_size = 30
    test_loader = get_livecell_loader(
        "/nfs/home/e7faffa3966db4c3/data", patch_shape, "test",
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
          dropout_rate = 0.1,
          masked_pretrain = False)
    
    model.load_state_dict(torch.load("checkpoints/livecell-boundary-model/best.pt")["model_state"])
    model = model.eval()
    model = model.cuda()
    loss = torch_em.loss.DiceLoss()
    cumLoss = 0.0
    for src, tgt in tqdm(test_loader):
        src = src.to(torch.device("cuda"))
        tgt = tgt.to(torch.device("cuda"))
        cumLoss += loss(model(src), tgt)
    cumLoss /= len(test_loader)
    print(f"UNETR Cum test score: {1 - cumLoss}")

if __name__ == '__main__':
    eval2DUnet()
    eval2DUnetR()