import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch-ssize", default=32)
args = parser.parse_args()
p = args.batch_ssize
print(p, type(p))
