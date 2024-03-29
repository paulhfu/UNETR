# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at(16,) * self.spatial_dims
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple
import math

import torch.nn as nn
import torch
import random

from UNETR.utils import trunc_normal_, get_2d_sincos_pos_embed
from UNETR.patchembedding_blocks import EmbeddingBlock, RectPatchingBlock
from UNETR.unetr_blocks import UnetResBlock, UnetrPrUpBlock, UnetrUpBlock, UnetrUpBlockNoSkip
from UNETR.unet_blocks import UnetOutBlock
from UNETR.vit import ViT


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    For this code, 
    https://github.com/facebookresearch/mae/tree/efb2a8062c206524e35e47d04501ed4f544c0ae8
    and
    https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV
    where used as starting points.
    This needs: https://github.com/paulhfu/torch-em.git
    export PYTHONPATH=$PYTHONPATH:/nfs/home/e7faffa3966db4c3/Documents/torch-em
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        conv_block: bool = False,
        dropout_rate: float = 0.0,
        masked_pretrain = False,
        masking_ratio = 0.75,
        patch_shape = "square"
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")
        assert patch_shape in ["square", "line"]
        if patch_shape == "square":
            for s in img_size:
                assert s % 16 == 0
        else:
            for s in img_size[:-1]:
                assert s % 32 == 0
        self.spatial_dims = len(img_size)
        self.num_layers = 12
        self.patch_size = (16,) * self.spatial_dims if patch_shape == "square" else (32, 8) * (self.spatial_dims - 1)
        self.patch_sqr_size = (16,) * self.spatial_dims
        self.feat_size = tuple((simg // spatch for simg, spatch in zip(img_size, self.patch_size)))
        self.n_patches = torch.tensor(self.feat_size).prod().item()
        self.hidden_size = hidden_size
        self.classification = False
        self.masked_pretrain = masked_pretrain
        self.masking_ratio = masking_ratio
        self.n_unmasked = round(self.n_patches * (1 - self.masking_ratio))
        #self.position_decoder_embed = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        #trunc_normal_(self.position_decoder_embed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.hidden_size), requires_grad=False)
            
        self.patch = RectPatchingBlock(
            img_size=img_size,
            patch_size=self.patch_size,
            spatial_dims=self.spatial_dims,
            patch_sqr_size=self.patch_sqr_size
        )
        self.n_patches = self.patch.n_patches
        self.patch_size = self.patch.patch_size
        self.patch_embedding = EmbeddingBlock(
            in_channels=in_channels,
            n_patches=self.n_patches,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.vit = ViT(
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.init_decoder(in_channels, feature_size, hidden_size, conv_block, out_channels, self.masked_pretrain)
        
    def init_decoder(self, in_channels, feature_size, hidden_size, conv_block, out_channels, masked_pretrain):
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            conv_block=conv_block
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            conv_block=conv_block
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            conv_block=conv_block
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2
        )
        
        if masked_pretrain:
            self.decoder2 = UnetrUpBlockNoSkip(
                spatial_dims=self.spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2
            )
            self.encoder1 = None
            self.mask_token = nn.Parameter(torch.zeros((1, 1, hidden_size)))
        else:
            self.decoder2 = UnetrUpBlock(
                spatial_dims=self.spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2
            )
            self.encoder1 = UnetResBlock(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=1
            )
            self.mask_token = None
        
        self.out = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def freeze_encoder(self):
        for param in self.vit.parameters():
            param.requires_grad = False
            
    def unfreeze_encoder(self):
         for param in self.vit.parameters():
            param.requires_grad = True    
               
    def disable_masking(self):
        self.masked_pretrain = False
        
    def enable_masking(self):
        self.masked_pretrain = True

    def forward(self, x_in):
        input_data = x_in.clone()
        x = self.patch(x_in)
        x = self.patch_embedding(x)
        mask = None  # in case of unsupervised pretraining, this will bee needed by the loss to mask out unmasked patches.
        if self.masked_pretrain:
            noise = torch.rand(x.shape[:2], device=x.device)
            mask = torch.ones(x.shape[:2], device=x.device)
            coin = random.randint(0, 1)
            noise[:, coin::2] += 1
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            ids_keep = ids_shuffle[:, :self.n_unmasked]
            x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, self.hidden_size))
            mask[:, :self.n_unmasked] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
        x, hidden_states_out = self.vit(x)
        x2 = hidden_states_out[3]
        x3 = hidden_states_out[6]
        x4 = hidden_states_out[9]
        if self.masked_pretrain:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x = torch.gather(torch.cat([x, mask_tokens], dim=1), dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x2 = torch.gather(torch.cat([x2, mask_tokens], dim=1), dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x2.shape[2]))
            x3 = torch.gather(torch.cat([x3, mask_tokens], dim=1), dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x3.shape[2]))
            x4 = torch.gather(torch.cat([x4, mask_tokens], dim=1), dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x4.shape[2]))
            #x = x + self.decoder_pos_embed
            enc2 = self.encoder2(self.patch.projectFeatures(x2))
            enc3 = self.encoder3(self.patch.projectFeatures(x3))
            enc4 = self.encoder4(self.patch.projectFeatures(x4))

            dec3 = self.decoder5(self.patch.projectFeatures(x), enc4)
            dec2 = self.decoder4(dec3, enc3)
            dec1 = self.decoder3(dec2, enc2)
            out = self.decoder2(dec1)
            logits = self.out(out)
            return [logits, mask, input_data, self.patch]
            
        enc1 = self.encoder1(x_in)
        enc2 = self.encoder2(self.patch.projectFeatures(x2))
        enc3 = self.encoder3(self.patch.projectFeatures(x3))
        enc4 = self.encoder4(self.patch.projectFeatures(x4))

        dec4 = self.patch.projectFeatures(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits