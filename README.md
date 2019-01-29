# realNVP
_A PyTorch implementation of the training procedure of [Density Estimation Using Real NVP](https://arxiv.org/pdf/1605.08803.pdf)_. The original implementation in TensorFlow can be found at <https://github.com/tensorflow/models/tree/master/research/real_nvp>. 

## Imlementation Details
This implementation supports training on four datasets, namely **CIFAR-10**, **CelebA**, **ImageNet 32x32** and **ImageNet 64x64**. For each dataset, only the training split is used for learning the distribution. Labels are left untouched. Raw data is subject to dequantization, random horizontal flipping and logit transformation (see the paper for details). The network architecture is faithfully reproduced. The same set of hyperparameters as suggested by the paper is set as default. Adam with default parameters are used for optimization. Model performance, evaluated by bits/dim, matches what was reported in the paper.

## Samples
The samples are generated from models trained with default parameters. Each iteration corresponds to a minibatch of 64 images.

**CIFAR-10**

_1000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/cifar10/bs64_normal_bd64_rb8_bn0_sk1_wn1_cb1_af1_1000.png?raw=true "CIFAR-10 1000 iterations")

_80000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/cifar10/bs64_normal_bd64_rb8_bn0_sk1_wn1_cb1_af1_80000.png?raw=true "CIFAR-10 80000 iterations")

**CelebA**

_1000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/celeba/bs64_normal_bd32_rb2_bn0_sk1_wn1_cb1_af1_1000.png?raw=true "CelebA 1000 iterations")

_60000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/celeba/bs64_normal_bd32_rb2_bn0_sk1_wn1_cb1_af1_60000.png?raw=true "CelebA 60000 iterations")

**ImageNet 32x32**

_1000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/imnet32/bs64_normal_bd32_rb4_bn0_sk1_wn1_cb1_af1_1000.png?raw=true "ImageNet 32x32 1000 iterations")

_80000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/imnet32/bs64_normal_bd32_rb4_bn0_sk1_wn1_cb1_af1_80000.png?raw=true "ImageNet 32x32 80000 iterations")

**ImageNet 64x64**

_1000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/imnet64/bs64_normal_bd32_rb2_bn0_sk1_wn1_cb1_af1_1000.png?raw=true "ImageNet 64x64 1000 iterations")

_60000 iterations_

![](https://github.com/fmu2/realNVP/blob/master/samples/imnet64/bs64_normal_bd32_rb2_bn0_sk1_wn1_cb1_af1_60000.png?raw=true "ImageNet 64x64 60000 iterations")

## Training

Code runs on a single GPU and has been tested with

- Python 3.7.2
- torch 1.0.0
- numpy 1.15.4

```
python train.py --dataset=cifar10 --batch_size=64 --base_dim=64 --res_blocks=8 --max_iter=80000
python train.py --dataset=celeba --batch_size=64 --base_dim=32 --res_blocks=2 --max_iter=60000
python train.py --dataset=imnet32 --batch_size=64 --base_dim=32 --res_blocks=4 --max_iter=80000
python train.py --dataset=imnet64 --batch_size=64 --base_dim=32 --res_blocks=2 --max_iter=60000 
```


