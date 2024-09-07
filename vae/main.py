import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import numpy as np
import json
import argparse
from train_eval_vae import train_eval

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="holdout", help="Data division mode. 'holdout' for holdout train/test separation; 'folds' for 10-fold split")
    args = parser.parse_args()
    with open('hyperparams.json', 'r') as f:
        grid = json.load(f)
    print(grid)
    if args.mode == "holdout":
        print("holdout")
    else:
        print("folds")

if __name__ == "__main__":
    main()