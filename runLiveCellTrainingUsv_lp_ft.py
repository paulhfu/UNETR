import sys
sys.path.insert(0, "/home/e7faffa3966db4c3/Documents/torch-em")
import torch
from UNETR.model import UNETR
from UNETR.mae_loss import MaeLoss
import torch_em
from torch_em.data.datasets import get_livecell_loader
from torch_em.trainer.tensorboard_logger import MaskedPretrainLogger

masked_pretrain = True
linear_probing = True
finetune = True

def train_boundaries():
    patch_shape = (512, 512)
    if masked_pretrain:
        batch_size = 10
        train_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "train",
            download=True, boundaries=True, batch_size=batch_size
        )
        val_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "val",
            boundaries=True, batch_size=batch_size
        )
        model = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=patch_shape,
            feature_size = 16,
            hidden_size = 768,
            mlp_dim = 3072,
            num_heads = 12,
            conv_block = True,
            dropout_rate = 0.1,
            masking_ratio = 0.4,
            masked_pretrain = True,
            patch_shape="line")
        
        loss = MaeLoss()
        
        trainer = torch_em.default_segmentation_trainer(
            name="livecell-mae",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss=loss,
            metric=loss,
            learning_rate=1e-4,
            device=torch.device("cuda"),
            mixed_precision=True,
            logger=MaskedPretrainLogger,
            log_image_interval=50
        )
        trainer.fit(iterations=100000)

    if linear_probing:
        batch_size = 5
        train_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "train",
            download=True, boundaries=True, batch_size=batch_size
        )
        val_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "val",
            boundaries=True, batch_size=batch_size
        )
        model = UNETR(
            in_channels=1,
            out_channels=1,
            img_size=patch_shape,
            feature_size = 16,
            hidden_size = 768,
            mlp_dim = 3072,
            num_heads = 12,
            conv_block = True,
            dropout_rate = 0.1,
            masking_ratio = 0.75,
            masked_pretrain = True)
        model.load_state_dict(torch.load("checkpoints/livecell-mae/best.pt")["model_state"])
        model.init_decoder(in_channels=1, feature_size=16, hidden_size=768, conv_block=True, out_channels=2)
        model.freeze_encoder()
        model.disable_masking()
        loss = torch_em.loss.DiceLoss()
        trainer = torch_em.default_segmentation_trainer(
            name="livecell-boundary-model-lp",
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
        trainer.fit(iterations=50000)

    if finetune:
        batch_size = 5
        train_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "train",
            download=True, boundaries=True, batch_size=batch_size
        )
        val_loader = get_livecell_loader(
            "/home/e7faffa3966db4c3/data",
            #"~/data",
            patch_shape, "val",
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
            masking_ratio = 0.75,
            masked_pretrain = False)
        model.load_state_dict(torch.load("checkpoints/livecell-boundary-model-lp/best.pt")["model_state"])
        model.unfreeze_encoder()
        loss = torch_em.loss.DiceLoss()
        trainer = torch_em.default_segmentation_trainer(
            name="livecell-boundary-model-ft",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss=loss,
            metric=loss,
            learning_rate=5e-5,
            device=torch.device("cuda"),
            mixed_precision=True,
            log_image_interval=50
        )
        trainer.fit(iterations=50000)


if __name__ == '__main__':
    train_boundaries()