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

"""https://openreview.net/pdf?id=IDwN6xjHnK8, Appendix A2

"SwinT-Hyperprior, SwinT-ChARM For both SwinT-Hyperprior and SwinT-ChARM, we use the
same configurations: (wg, wh) = (8, 4), (C1, C2, C3, C4, C5, C6) = (128, 192, 256, 320, 192, 192),
(d1, d2, d3, d4, d5, d6) = (2, 2, 6, 2, 5, 1) where C, d, and w are defined in Figure 13 and Figure 2.
The head dim is 32 for all attention layers in SwinT-based models."

Note: It is not clear from Figure 2 whether d1-d6 are applied symmetrically to both encoder and decoder;
in our implementation, we set the depth for Ga, Ha to 1, i.e. no SW-MSA is applied on the encoder side.
-> update as required.
"""

class ConfigGa:
  embed_dim = [128, 192, 256, 320]
  embed_out_dim = [192, 256, 320, None]
  depths = [1, 1, 1, 1]
  #depths = [2, 2, 6, 2]
  head_dim = [32, 32, 32, 32]
  window_size = [8, 8, 8, 8]
  num_layers = len(depths)

class ConfigHa:
  embed_dim = [192, 192]
  embed_out_dim = [192, None]
  depths = [1, 1]
  # depths = [5, 1]
  head_dim = [32, 32]
  window_size = [4, 4]
  num_layers = len(depths)

class ConfigHs:
  embed_dim = [192, 192]
  embed_out_dim = [192, 320]
  depths = [1, 5]
  head_dim = [32, 32]
  window_size = [4, 4]
  num_layers = len(depths)

class ConfigGs:
  embed_dim = [320, 256, 192, 128]
  embed_out_dim = [256, 192, 128, 3]
  depths = [2, 6, 2, 2]
  head_dim = [32, 32, 32, 32]
  window_size = [8, 8, 8, 8]
  num_layers = len(depths)