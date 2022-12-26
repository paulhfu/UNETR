# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from UNETR.utils import trunc_normal_

SUPPORTED_EMBEDDING_TYPES = {"conv", "perceptron"}
        
class RectPatchingBlock(nn.Module):
    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        patch_sqr_size: Sequence[int],
        spatial_dims: int = 3,
    ):
        super().__init__()
        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
        self.patch_size = patch_size
        self.n_patches = [im_d // p_d for im_d, p_d in zip(img_size, patch_size)]
        self.n_sqrPatches = [im_d // p_d for im_d, p_d in zip(img_size, patch_sqr_size)]
        chars = (("h", "p1"), ("w", "p2"), ("d", "p3"))[:spatial_dims]
        from_chars = "b c " + " ".join(f"({k} {v})" for k, v in chars)
        to_chars = f"b ({' '.join([c[0] for c in chars])}) ({' '.join([c[1] for c in chars])} c)"
        axes_len = {f"p{i+1}": p for i, p in enumerate(self.patch_size)}
        self.patches = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)
        self.patchesInverse = Rearrange(f"{to_chars} -> {from_chars}", h=self.n_patches[0], w=self.n_patches[1], **axes_len)
        axes_len = {f"p{i+1}": p for i, p in enumerate(patch_sqr_size)}
        to_chars = f"b {' '.join([c[0] for c in chars])} ({' '.join([c[1] for c in chars])} c)"
        self.patchesRect = Rearrange(f"{from_chars} -> {to_chars}", **axes_len)
        self.n_patches = np.prod(self.n_patches)
        
    def forward(self, x):
        x = self.patches(x)
        return x

    def inversePatching(self, x):
        x = self.patchesInverse(x)
        return x
    
    def projectFeatures(self, x):
        x = self.patchesInverse(x)
        x = self.patchesRect(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class EmbeddingBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_patches: int,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden size should be divisible by num_heads.")

        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")
        self.n_patches = n_patches
        self.patch_dim = int(in_channels * np.prod(patch_size))
        self.embeddings = nn.Linear(self.patch_dim, hidden_size)

        #convCtor = nn.Conv3d if spatial_dims == 3 else nn.Conv2d
        #self.patch_embeddings = convCtor(in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size)
        # for 3d: "b c (h p1) (w p2) (d p3)-> b (h w d) (p1 p2 p3 c)"

        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))  # Prob not needed
        self.dropout = nn.Dropout(dropout_rate)
        trunc_normal_(self.position_embeddings, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.embeddings(x)
        #x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings