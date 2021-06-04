# PyTorch SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

This repository is a simplified version of original [SimCLR repository](https://github.com/sthalles/SimCLR/tree/e8a690ae4f4359528cfba6f270a9226e3733b7fa)

We are trying to show an inconsitent batch size leads error for some training cases. If you uncomment Line 295 of [resnet_train.py](./resnet_train.py), the code will work correctly.

Also, there is a chanllenging part that requires exact values of tensor to infer the shape at Line 216. Temporarily, we altered it to random tensor for this experiment.

## How to test

```bash
python bin/pytea.py experiment/SimCLR
```
