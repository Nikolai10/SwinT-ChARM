# SwinT-ChARM (TensorFlow 2)

[<img src="https://colab.research.google.com/assets/colab-badge.svg" align="center">](TBD: Version 2.0)

This repository provides a TensorFlow implementation of SwinT-ChARM based on:

- [Transformer-Based Transform Coding (ICLR 2022)](https://openreview.net/pdf?id=IDwN6xjHnK8),
- [Channel-wise Autoregressive Entropy Models For Learned Image Compression (ICIP 2020)](https://arxiv.org/pdf/2007.08739.pdf).

![SwinT-ChARM net arch](https://github.com/Nikolai10/SwinT-ChARM/blob/master/res/doc/figures/teaser.png)
<sup>
[Source](https://openreview.net/pdf?id=IDwN6xjHnK8)
</sup>

## Updates

***10/09/2022***

1. The number of model parameters now corresponds exactly to the reported number (32.6 million). We thank the authors for providing us with the official DeepSpeed log files.
2. SwinT-ChARM now supports compression at different input resolutions (multiples of 256).
3. We release a pre-trained model as proof of functional correctness.

***08/17/2022***

1. Initial release of this project (see branch *release_08/17/2022*)

## Acknowledgment
This project is based on:

- [TensorFlow Compression (TFC)](https://github.com/tensorflow/compression), a TF library dedicated to data compression.
- [swin-transformers-tf](https://github.com/sayakpaul/swin-transformers-tf), an unofficial implementation of [Swin-Transformer](https://github.com/microsoft/Swin-Transformer). Functional correctness has been [proven](https://github.com/microsoft/Swin-Transformer/pull/206).

Note that this repository builds upon the official TF implementation of [Minnen et al.](https://github.com/tensorflow/compression/blob/master/models/ms2020.py), while Zhu et al. base their work on an
unknown (possibly not publicly available) PyTorch reimplementation.

## Example (kodim22)

Original | SwinT-ChARM (β = 0.0003)
:-------------------------:|:-------------------------:
![kodim22.png](https://github.com/Nikolai10/SwinT-ChARM/blob/master/res/eval/kodim22.png) | ![kodim22_hat.png](https://github.com/Nikolai10/SwinT-ChARM/blob/master/res/eval/kodim22_hat.png)


```python
Mean squared error: 13.7772
PSNR (dB): 36.74
Multiscale SSIM: 0.9871
Multiscale SSIM (dB): 18.88
Bits per pixel: 0.9890
```

## Pretrained Models

| Lagrangian multiplier (β) | link |
| ----------- | -------------------------------- | 
| 0.0003 | [TBD](...) |

## File Structure

     res
         ├── doc/                                       # addtional resources
         ├── eval/                                      # sample images + reconstructions
         ├── train_zyc2022/                             # model checkpoints + tf.summaries
         ├── zyc2022/                                   # saved model
     swin-transformers-tf/                              # extended swin-transformers-tf implementation 
         ├── changelog.txt                              # summary of changes made to the original work
         ├── ...  
     config.py                                          # model-dependent configurations
     zyc2022.py                                         # core of this repo

## License
[Apache License 2.0](LICENSE)
