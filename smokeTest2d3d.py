from UNETR.model import UNETR

import torch

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
        masked_pretrain = True)

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
        masked_pretrain = True)

out = model(torch.normal(mean=torch.ones(size=(1, 1, 64, 64))))[0].sum()
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