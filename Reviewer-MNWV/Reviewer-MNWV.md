| Method    | Architecture | Accuracy±std   | Time per Epoch |
| --------- | ------------ | -------------- | -------------- |
| DivideMix | ResNet-50    | 74.59±0.55     | 717s           |
| IDO       | ResNet-50    | **74.77±0.48** | **229s**       |

<center>Table 1. The results of DivideMix and IDO with the setting of pre-trained ResNet50, with SGD, lr = 2e-3, weight_decay = 1e-3, momentum=0.9, batch_size=64. One epoch has 1000 iterations, and 100 epochs are trained. We set stage 1 lasts 2 epochs, stage 2 lasts 98 epochs. The experiment was performed on a single A100 80GB, repeated 5 times</center>


| Noise     | Architecture   | Sym. 20%     | Sym. 40%     | Sym. 60%     | Asym. 40%    | Inst. 40%    |
| --------- | -------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| Standard  | ResNet-50      | 93.2%        | 92.3%        | 88.2%        | 91.1%        | 90.9%        |
| UNICON    | ResNet-50      | 94.8%        | 93.2%        | 92.5%        | 93.5%        | 93.9%        |
| ELR       | ResNet-50      | 96.5%        | 95.8%        | 95.1%        | <u>95.2%</u> | 94.8%        |
| DeFT      | CLIP-ResNet-50 | 96.9%        | <u>96.6%</u> | 95.7%        | 93.8%        | 95.1%        |
| DivideMix | ResNet-50      | <u>97.1%</u> | **96.9%**    | <u>96.3%</u> | 93.1%        | 96.0%        |
| DISC      | ResNet-50      | 96.8%        | 96.5%        | 95.5%        | 95.1%        | **96.5%**    |
| IDO       | ResNet-50      | **97.3%**    | **96.9%**    | **96.5%**    | **95.3%**    | <u>96.4%</u> |

<center>Table 2. The results of Standard, UNICON, ELR, DeFT, DivideMix, and IDO on CIFAR-10 with five different noise levels. The experiment setting is followed CIFAR-100 setting in our paper. The experiment was performed on a single A100 80GB, repeated 5 times</center>



