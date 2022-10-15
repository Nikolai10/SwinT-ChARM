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

import collections.abc
from functools import partial
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

from ..layers import StochasticDepth, WindowAttention
from . import utils
from .mlp import mlp_block


class SwinTransformerBlock(keras.Model):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        window_size (int): Window size.
        num_heads (int): Number of attention heads.
        head_dim (int): Enforce the number of channels per head
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        norm_layer (layers.Layer, optional): Normalization layer.  Default: layers.LayerNormalization
    """

    def __init__(
            self,
            dim,
            num_heads=4,
            head_dim=None,
            window_size=7,
            shift_size=0,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert (
                0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"
        self.norm1 = norm_layer()
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size
            if isinstance(window_size, collections.abc.Iterable)
            else (window_size, window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            name="window_attention",
        )
        self.drop_path = (
            StochasticDepth(drop_path) if drop_path > 0.0 else tf.identity
        )
        self.norm2 = norm_layer()
        self.mlp = mlp_block(
            dropout_rate=drop, hidden_units=[int(dim * mlp_ratio), dim]
        )
        self.attn_mask = None

    def get_img_mask(self):
        # calculate image mask for SW-MSA
        # since Tensorflow does not support item assignment, we use a
        # "hacky" solution. See
        # https://github.com/microsoft/Swin-Transformer/blob/e43ac64ce8abfe133ae582741ccaf6761eea05f7/models/swin_transformer.py#L222
        # for more information.

        H, W = self.input_resolution
        window_size = self.window_size

        mask_0 = tf.zeros((1, H-window_size, W-window_size, 1))
        mask_1 = tf.ones((1, H-window_size, window_size//2, 1))
        mask_2 = tf.ones((1, H-window_size, window_size//2, 1))
        mask_2 = mask_2+1
        mask_3 = tf.ones((1, window_size//2, W-window_size, 1))
        mask_3 = mask_3+2
        mask_4 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_4 = mask_4+3
        mask_5 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_5 = mask_5+4
        mask_6 = tf.ones((1, window_size//2, W-window_size, 1))
        mask_6 = mask_6+5
        mask_7 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_7 = mask_7+6
        mask_8 = tf.ones((1, window_size//2, window_size//2, 1))
        mask_8 = mask_8+7

        mask_012 = tf.concat([mask_0, mask_1, mask_2], axis=2)
        mask_345 = tf.concat([mask_3, mask_4, mask_5], axis=2)
        mask_678 = tf.concat([mask_6, mask_7, mask_8], axis=2)

        img_mask = tf.concat([mask_012, mask_345, mask_678], axis=1)
        return img_mask


    def get_attn_mask(self):
        # calculate attention mask for SW-MSA
        mask_windows = utils.window_partition(
            self.img_mask, self.window_size
        )  # [num_win, window_size, window_size, 1]
        mask_windows = tf.reshape(
            mask_windows, (-1, self.window_size * self.window_size)
        )
        attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(
            mask_windows, 2
        )
        attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
        return tf.where(attn_mask == 0, 0.0, attn_mask)

    def call(
            self, x, return_attns=False
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:

        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        self.input_resolution = (H, W)

        if self.shift_size > 0:
            self.img_mask = tf.stop_gradient(self.get_img_mask())
            self.attn_mask = self.get_attn_mask()

        x = tf.reshape(x, (-1, H*W, C))

        shortcut = x
        x = self.norm1(x)
        x = tf.reshape(x, (-1, H, W, C))

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = utils.window_partition(
            shifted_x, self.window_size
        )  # [num_win*B, window_size, window_size, C]
        x_windows = tf.reshape(
            x_windows, (-1, self.window_size * self.window_size, C)
        )  # [num_win*B, window_size*window_size, C]

        # W-MSA/SW-MSA
        if not return_attns:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # [num_win*B, window_size*window_size, C]
        else:
            attn_windows, attn_scores = self.attn(
                x_windows, mask=self.attn_mask, return_attns=True
            )  # [num_win*B, window_size*window_size, C]
        # merge windows
        attn_windows = tf.reshape(
            attn_windows, (-1, self.window_size, self.window_size, C)
        )
        shifted_x = utils.window_reverse(
            attn_windows, self.window_size, H, W
        )  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x,
                shift=(self.shift_size, self.shift_size),
                axis=(1, 2),
            )
        else:
            x = shifted_x

        x = tf.reshape(x, (-1, H * W, C))

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = tf.reshape(x, (-1, H, W, C))

        if return_attns:
            return x, attn_scores
        else:
            return x