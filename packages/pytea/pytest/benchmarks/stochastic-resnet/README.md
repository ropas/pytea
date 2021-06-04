# Noisy Student Self-Training on CIFAR-10 with Stochastic ResNet

This project implemented several machine learning techniques, especially stochastic network.

We cutoff some redundant codes (e.g. training students) for the purpose of showcase. Also, we set the length of cifar10 to 30/10 explicitly, because of its complex data manipulation path. See [config](./pyteaconfig.json).

## How to test

```bash
python bin/pytea.py experiment/stochastic-resnet
```

## References

1. Xiaojin Zhu. Semi-supervised learning. tutorial, 2007.
2. Qizhe Xie, Minh-Thang Luong, Eduard Hovy, and Quoc V. Le. Self-training with noisy student improves imagenet classification, 2020.
3. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015.
4. Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V. Le. Randaugment: Practical automated data augmentation with a reduced search space, 2019.
5. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Weinberger. Deep networks with stochastic depth, 2016.
6. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision, 2015.
7. Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization, 2018.
8. Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks, 2017.
9. Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples, 2015.
10. Morgane Goibert and Elvis Dohmatob. Adversarial robustness via label-smoothing, 2019.
11. Dan Hendrycks and Thomas G. Dietterich. Benchmarking neural network robustness to common corruptions and surface variations, 2019.

### Code References & Data

- RandAugment: [ildoonet/pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)
- Stochastic Depth on ResNet: [shamangary/Pytorch-Stochastic-Depth-Resnet](https://github.com/shamangary/Pytorch-Stochastic-Depth-Resnet)
- Mixup: [facebookresearch/mixup-cifar-10](https://github.com/facebookresearch/mixup-cifar10)
- CIFAR-10-C and CIFAR-10-P: [hendrycks/robustness](https://github.com/hendrycks/robustness)
