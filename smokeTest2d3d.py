from UNETR.model import UNETR
from UNETR.mae_loss import MaeLoss

import torch

loss = MaeLoss(3, 16)
model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(64, 64, 64),
        feature_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_heads = 12,
        conv_block = True,
        dropout_rate = 0.0,
        masked_pretrain = True)

out = loss(model(torch.normal(mean=torch.ones(size=(1, 1, 64, 64, 64)))), None)
out.backward()
model.init_decoder(in_channels=1, feature_size=16, hidden_size=768, conv_block=True, out_channels=2)
model.freeze_encoder()
model.disable_masking()
model.unfreeze_encoder()

loss = MaeLoss(2, 16)
model = UNETR(
        in_channels=1,
        out_channels=1,
        img_size=(64, 64),
        feature_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_heads = 12,
        conv_block = True,
        dropout_rate = 0.0,
        masked_pretrain = True)

out = loss(model(torch.normal(mean=torch.ones(size=(1, 1, 64, 64)))), None)
out.backward()

model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(64, 64, 64),
        feature_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_heads = 12,
        conv_block = True,
        dropout_rate = 0.0,
        masked_pretrain = False)

out = model(torch.normal(mean=torch.ones(size=(1, 1, 64, 64, 64))))[0].sum()
out.backward()

model = UNETR(
        in_channels=1,
        out_channels=2,
        img_size=(64, 64),
        feature_size = 16,
        hidden_size = 768,
        mlp_dim = 3072,
        num_heads = 12,
        conv_block = True,
        dropout_rate = 0.0,
        masked_pretrain = False)

out = model(torch.normal(mean=torch.ones(size=(1, 1, 64, 64))))[0].sum()
out.backward()