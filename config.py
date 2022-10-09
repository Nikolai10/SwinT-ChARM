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

"""

class ConfigGa:
  embed_dim = [128, 192, 256, 320]
  embed_out_dim = [192, 256, 320, None]
  depths = [2, 2, 6, 2]
  head_dim = [32, 32, 32, 32]
  window_size = [8, 8, 8, 8]
  num_layers = len(depths)

class ConfigHa:
  embed_dim = [192, 192]
  embed_out_dim = [192, None]
  depths = [5, 1]
  head_dim = [32, 32]
  window_size = [4, 4]
  num_layers = len(depths)

class ConfigHs:
  embed_dim = [192, 192]
  embed_out_dim = [192, int(2*320)]
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

"""https://arxiv.org/pdf/2007.08739.pdf, Appendix A

"To account for the different input depths, each CC [...] transform is 
programmatically defined to linearly interpolate between the input and the
output depth.

Note: In SwinT-ChARM a slightly different logic is used, which is why the depth 
is explicitly hardcoded here.

For example, the tenth slice should have depths: 224, 128 and 32. 

import pandas as pd
import numpy as np
a=pd.Series([320, np.nan, np.nan, 32])

a.interpolate(method='linear')

vs. 234, 117, 32 (taken from the official deepspeed logfile, 
which was provided by the authors).

"""

class ConfigChARM:
  depths_conv0 = [64, 64, 85, 106, 128, 149, 170, 192, 213, 234]
  depths_conv1 = [32, 32, 42, 53, 64, 74, 85, 96, 106, 117]