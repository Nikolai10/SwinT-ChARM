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

from functools import partial

import tensorflow as tf
from tensorflow.keras import layers as L


class PatchSplitting(L.Layer):
    """Patch Splitting Layer as described in 
    https://openreview.net/pdf?id=IDwN6xjHnK8 (section 3.1)
    # Patch Split = [Linear, LayerNorm, Depth-to-Space (for upsampling)]
    Args:
        dim (int): Number of input channels.
    """

    def __init__(
            self,
            dim,
            out_dim=None,
            norm_layer=partial(L.LayerNormalization, epsilon=1e-5),
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.out_dim = out_dim or dim
        self.norm = norm_layer()
        self.reduction = L.Dense(self.out_dim * 4, use_bias=False)

    def call(self, x):
        """
        x: B, H, W, C
        """
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (-1, H * W, C))

        x = self.reduction(x)
        x = self.norm(x)

        x = tf.reshape(x, (-1, H, W, self.out_dim * 4))
        x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
        x = tf.reshape(x, (-1, 2 * H, 2 * W, self.out_dim))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "out_dim": self.out_dim,
                "norm": self.norm,
            }
        )
        return config


class PatchUnpack(L.Layer):
    """Patch Unpack Layer
    # PatchUnpack = [Linear, Depth-to-Space (for upsampling)]

    Key differences to PatchSplitting:
    - no LayerNorm
    - use_bias=True (self.reduction)

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(
            self,
            dim,
            out_dim=None,
            norm_layer=None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.out_dim = out_dim or dim
        self.reduction = L.Dense(self.out_dim * 4, use_bias=True)

    def call(self, x):
        """
        x: B, H, W, C
        """
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = tf.reshape(x, (-1, H * W, C))

        x = self.reduction(x)

        x = tf.reshape(x, (-1, H, W, self.out_dim * 4))
        x = tf.nn.depth_to_space(x, 2, data_format='NHWC')
        x = tf.reshape(x, (-1, 2 * H, 2 * W, self.out_dim))

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dim": self.dim,
                "out_dim": self.out_dim,
            }
        )
        return config
