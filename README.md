# Layer-level activation (LayerAct) Mechanism
This repository contains the PyTorch based implementation of Layer-level activation (LayerAct) functions. 

## Enviornment and Datasets 
### Install
Create a conda virtual environment and activate it. 
```
conda create -n layeract python=3.9.12
conda activate layeract
```

Install `PyTorch` and `torchvision` with `CUDA`. 
```
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
```

Install other requirements. 
```
pip install scikit-learn==1.0.2 pandas==2.1.0 mpmath==1.3.0
```

### Data preparation
For the experimental reproduction, you need 7 datasets, (CIFAR10, CIAFR100, ImageNet, CIFAR10-C, CIFAR100-C, ImageNet-C, and medical image) in total. 
- For the CIFAR10 and CIFAR100 datasets, visit the [official website](https://www.cs.toronto.edu/~kriz/cifar.html) or use [torchvision datasets](https://pytorch.org/vision/main/datasets.html). 
- For the ImageNet dataset, visit the [official website](https://www.image-net.org/challenges/LSVRC/2012/index.php#) or use [torchvision datasets](https://pytorch.org/vision/main/datasets.html).
- For the CIFAR10-C, CIFAR100-C, and ImageNet-C datasets, visit the [official github repository of Hendrycks, D. and Dietterich, T. (2019)](https://github.com/hendrycks/robustness?tab=readme-ov-file).
- For the medical image dataset, visit the [2018 Data Science Bowl](https://www.kaggle.com/competitions/data-science-bowl-2018).

## Usage 
### Evaluation 
We provide the trained networks from our experiments are available on our [anonymous Google Drive](https://drive.google.com/drive/folders/10LNLxGxyDVCk1J3wthxmCYsyKnpC6CC1?usp=sharing).
- Use the scripts file in the `scripts` folder to evaluate the trained networks.

- For example, to evaluate the ResNet20 with LA-SiLU on CIFAR10 with a single GPU (cuda:0):
```
bash ./scripts/CIFAR/test.sh CIFAR10 resnet20 la_silu <your-CIFAR10-path> <your-GPU-device-number-to-use>
```

- To evaluate the networks on CIFAR-C datasets (CIFAR10-C and CIFAR100-C):
```
bash ./scripts/CIFAR/c_test.sh CIFAR10 resnet20 la_silu <your-CIFAR10-path> <your-GPU-device-number-to-use>
```

- To evaluate the networks on ImageNet-C datasets:
```
bash ./scripts/ImageNet/c_test.sh resnet50 la_silu <your-CIFAR10-path> <your-GPU-device-number-to-use>
```

### Training a new network 
We trained the networks with one GPU device and eight GPU devices for CIFAR10, CIFAR100, and medical image dataset, and ImageNet, respectively. 

- To train ResNets (`resnet20`, `resnet32`, and  `resnet44`) on CIFAR datasets, run:
```
bash ./scripts/CIFAR/train.sh CIFAR10 resnet20 la_silu <your-CIFAR10-path> <your-GPU-device-id-to-use>
```

- To train ResNets (`resnet50` and `resnet101`) on ImageNet dataset, run:
```
bash ./scripts/ImageNet/train.sh resnet50 la_silu <your-CIFAR10-path> <your-GPU-device-ids-to-use>
```

- To train UNets on medical image dataset, run:
```
bash ./scripts/UNet/train.sh la_silu <your-CIFAR10-path> <your-GPU-device-ids-to-use>
```
