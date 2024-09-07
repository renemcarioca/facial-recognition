import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, utils
from sklearn.model_selection import train_test_split
import numpy as np
import json
import argparse
from train_eval_vae import train_eval
from PIL import Image



SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
dataset_path = '../lfw/'

BATCH_SIZE = 16
g = torch.Generator()
g.manual_seed(SEED)


def extract_pairs_for_development(pairs_filepath: str):
    pairs = []
    with open(pairs_filepath, 'r') as f:
        size = int(next(f))
        for i in range(size):
            line = next(f)
            name, num1, num2 = line.strip().split()
            pairs.append(((name, int(num1)), (name, int(num2))))
        for i in range(size):
            line = next(f)
            name1, num1, name2, num2 = line.strip().split()
            pairs.append(((name1, int(num1)), (name2, int(num2))))
    return pairs

class PairDataGenerator(Dataset):
    def __init__(self, pairs, transform=None):
        self.pairs = pairs

        composition = []
        if transform:
            composition.append(transform)
        composition.append(transforms.ToTensor())

        self.transform = transforms.Compose(composition)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        (name1, num1), (name2, num2) = self.pairs[index]
        # Assuming images are in a folder named 'lfw-deepfunneled'
        image_path1 = f'{dataset_path}/{name1}/{name1}_{num1:0>4}.jpg'
        image_path2 = f'{dataset_path}/{name2}/{name2}_{num2:0>4}.jpg'

        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        y_true = 1 if name1 == name2 else 0
        return image1, image2, y_true


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="dev", help="Data division mode. 'dev' for development holdout train/test separation; 'folds' for 10-fold split")
    args = parser.parse_args()
    with open('hyperparams.json', 'r') as f:
        grid = json.load(f)
    size_grid = grid['size']
    z_dim_grid = grid['z_dim']
    conv_blocks_grid = grid['conv_blocks']
    factor_grid = grid['r_loss_factor']
    print(grid)
    if args.mode == "dev":
        pairs_train_path = '../pairsDevTrain.txt'
        pairs_test_path = '../pairsDevTest.txt'
        pairs_train = extract_pairs_for_development(pairs_train_path)
        pairs_test = extract_pairs_for_development(pairs_test_path)
        hyperparam_list = []
        accuracy_list = []
        for size in size_grid:
            shape = (size, size)
            transform = transforms.Resize(shape)
            full_train_pair_dataset = PairDataGenerator(pairs_train, transform)

            full_train_pair_y = np.array([x[2] for x in full_train_pair_dataset])
            index = np.arange(len(full_train_pair_y))
            train_index, val_index, _, _ = train_test_split(index, full_train_pair_y, test_size=.2, stratify=full_train_pair_y, random_state=42)

            train_pair_dataset = Subset(full_train_pair_dataset, train_index)
            val_pair_dataset = Subset(full_train_pair_dataset, val_index)
                        
            train_pair_loader = DataLoader(train_pair_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
            val_pair_loader = DataLoader(val_pair_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
            
            for z_dim in z_dim_grid:
                for conv_blocks in conv_blocks_grid:
                    for factor in factor_grid:
                        hp_dict = {
                            "size" : size,
                            "z_dim" : z_dim,
                            "conv_blocks" : conv_blocks,
                            "r_loss_factor" : factor                             
                        }
                        hyperparam_list.append(hp_dict)
                        routine_tag = "devHPO"
                        val_acc = train_eval(train_pair_loader,
                                             val_pair_loader,
                                             hp_dict,
                                             routine_tag)[1]
                        accuracy_list.append(val_acc)
        print(hyperparam_list)
        print(accuracy_list)
        best = np.argmax(np.array(accuracy_list))
        print(hyperparam_list[best])
        print(accuracy_list[best])        


if __name__ == "__main__":
    main()