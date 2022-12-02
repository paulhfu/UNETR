import torch
from UNETR.model import UNETR
from UNETR.mae_loss import mae_loss
import torch_em
from torch_em.data.datasets import get_livecell_loader


def train_boundaries():
    n_out = 2
    #patch_shape = (512, 512)
    patch_shape = (64, 64)
    batch_size = 1
    #patch_shape = (384, 384)
    #batch_size = 3
    model = UNETR(
          in_channels=1,
          out_channels=n_out,
          img_size=patch_shape,
          feature_size = 16,
          hidden_size = 768,
          mlp_dim = 3072,
          num_heads = 12,
          conv_block = True,
          dropout_rate = 0.1,
          masked_pretrain = False)

    train_loader = get_livecell_loader(
        #"/nfs/home/e7faffa3966db4c3/data", 
        "~/data/images",
        patch_shape, "train",
        download=True, boundaries=True, batch_size=batch_size
    )
    val_loader = get_livecell_loader(
        #"/nfs/home/e7faffa3966db4c3/data", 
        "~/data/images",
        patch_shape, "val",
        boundaries=True, batch_size=batch_size
    )
    loss = mae_loss

    trainer = torch_em.default_segmentation_trainer(
        name="livecell-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=1)

    model.reinit_decoder()
    model.freeze_encoder()
    model.disable_masking()
    loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-4,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=1)

    model.unfreeze_encoder()
    loss = torch_em.loss.DiceLoss()
    trainer = torch_em.default_segmentation_trainer(
        name="livecell-boundary-model",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss=loss,
        metric=loss,
        learning_rate=1e-5,
        device=torch.device("cuda"),
        mixed_precision=True,
        log_image_interval=50
    )
    trainer.fit(iterations=1)


if __name__ == '__main__':
    train_boundaries()