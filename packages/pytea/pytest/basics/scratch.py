import argparse
import torch
from torchvision.models import resnet50

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

batch = args.batch_size
input = torch.rand(batch, 3, 224, 224)
model = resnet50()
output = model(input)

# fail
diff = output - torch.rand(batch, 10)

