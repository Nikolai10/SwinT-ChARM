# Copyright 2022 Nikolai Körber. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Code copied and modified from 
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer.py
"""

from functools import partial
from typing import Dict, Union
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from .swin_transformer_block import SwinTransformerBlock


class BasicLayer(keras.Model):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        head_dim (int): Channels per head (dim // num_heads if not set)
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | list[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer. Default: layers.LayerNormalization
        downsample (layers.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        upsample (layers.Layer | None, optional): Upsample layer at the end of the layer. Default: None
    """

    def __init__(
            self,
            dim,
            out_dim,
            depth,
            num_heads=4,
            head_dim=None,
            window_size=7,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
            downsample=None,
            upsample=None,
            **kwargs,
    ):

        super().__init__(kwargs)
        self.dim = dim
        self.depth = depth

        # build blocks
        blocks = [
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list)
                else drop_path,
                norm_layer=norm_layer,
                name=f"swin_transformer_block_{i}",
            )
            for i in range(depth)
        ]
        self.blocks = blocks
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None
        # patch splitting layer
        if upsample is not None:
            self.upsample = upsample(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
        else:
            self.upsample = None
    def call(
            self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if return_attns:
            attention_scores = {}

        for i, block in enumerate(self.blocks):
            if not return_attns:
                x = block(x)
            else:
                x, attns = block(x, return_attns)
                attention_scores.update({f"swin_block_{i}": attns})
        if self.downsample is not None:
            x = self.downsample(x)

        if self.upsample is not None:
            x = self.upsample(x)

        if return_attns:
            return x, attention_scores
        else:
            return x