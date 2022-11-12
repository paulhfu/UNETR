# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch.nn as nn

from UNETR.unetr_blocks import UnetResBlock, UnetrPrUpBlock, UnetrUpBlock
from UNETR.unet_blocks import UnetOutBlock
from UNETR.vit import ViT


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
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
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        self.spatial_dims = len(img_size)
        self.num_layers = 12
        self.patch_size = (16,) * self.spatial_dims
        self.feat_size = tuple((simg // spatch for simg, spatch in zip(img_size, self.patch_size)))
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            classification=self.classification,
            dropout_rate=dropout_rate,
        )
        self.encoder1 = UnetResBlock(
            spatial_dims=self.spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1
        )
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
        self.decoder2 = UnetrUpBlock(
            spatial_dims=self.spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2
        )
        self.out = UnetOutBlock(spatial_dims=self.spatial_dims, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        if self.spatial_dims == 3:
            x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
            x = x.permute(0, 4, 1, 2, 3).contiguous()
        else:
            x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits