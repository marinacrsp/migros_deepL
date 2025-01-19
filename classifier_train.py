import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import argparse
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from PIL import Image
import classifier_utils as cu
import dataset_utils as du
import sys
import os
import pandas as pd
import shutil


class ImageDataset(Dataset):
    def __init__(self, root_dir, df, size = 224, center_crop = True, train=True):
        self.root_dir = root_dir
        self.files = df['file_name'].tolist()
        self.labels = df['text'].tolist() 
        
        transforms_list = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        # Additional transformations for training
        if train:
            transforms_list.insert(3, transforms.RandomHorizontalFlip(p=0.5))
            transforms_list.insert(4, transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)))

        self.image_transforms = transforms.Compose(transforms_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        example = {}
        instance_image = Image.open(
            os.path.join(self.root_dir, self.files[idx])
        ).convert("RGB")

        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_labels"] = int(self.labels[idx])

        return example    

def get_data(data_train, data_test, batch_size=64):
    # CIFAR10 training dataset.
    
    metadata_train = pd.read_csv(data_train + '/metadata.csv')
    # if merge is true this is already done dealt with using the dataset utils, when the merged folder is created the csv is already adapted
    if merge == False:
        metadata_train['text'] = metadata_train['text'].apply(lambda x: 0 if 'healthy' in str(x).lower() else 1) # convert prompts to binary labels

    metadata_test = pd.read_csv(data_test + '/metadata.csv')
    metadata_test['text'] = metadata_test['text'].apply(lambda x: 0 if 'healthy' in str(x).lower() else 1) # convert prompts to binary labels
    
    print(f'\nTrain set size: {len(metadata_train)}, \nTest set size: {len(metadata_test)}')

    dataset_train = ImageDataset(
    root_dir=data_train,
    df=metadata_train
    )
    dataset_test = ImageDataset(
    root_dir=data_test,
    df=metadata_test
    )

    # Create data loaders.
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False) 
    return train_loader, test_loader


def check_and_make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        os.rmdir(path)
        os.mkdir(path)

###############################################################################################
#### Parser arguments
###############################################################################################
parser = argparse.ArgumentParser(description='Computing the AUC')
parser.add_argument('--test', type=str, help='Path to the folder containing real data for testing', required=True)
parser.add_argument('--train', type=str, help='Path to the folder containing real data for training', required=True)
parser.add_argument('--syn', type=str, help='Path to the folder containing synthetic data to be appended to the train', required=True)
parser.add_argument('--merge', type=bool, help='Specify whether to augment training data (i.e. use the syn argument or not)', required=True)
parser.add_argument('--merge_path', type=str, help='Path to where the merged train+synthetic folder is gonna be created (if merge=True)', required=True)
parser.add_argument('--epochs', type=int, help='N# of training epochs', required=False, default = 50)
parser.add_argument('--batch_size', type=int, help='Size of training batch', required=False, default = 32)
parser.add_argument('--lr', type=float, help='Learning rate', required=False, default = 1e-4)
parser.add_argument('--seed', type=int, help='Random seed', required=False, default = 21)
parser.add_argument('--verbose', type=bool, help='Set to true if additional information should be print along the process', required=False, default = True)
args = parser.parse_args()


dataset_train_path = args.train
dataset_test_path = args.test
dataset_generated_path = args.syn 
merge_path = args.merge_path
merge = args.merge
epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
verbose = args.verbose

seed = args.seed
if verbose:
    print('Random Seed Used: ', seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if verbose:
    print('DEVICE: ', device)

if verbose:
    print('ARGS: ', args)


if __name__ == "__main__":
    # First combine training data + synthetic if needed
    if merge:
        folder_merged = merge_path + 'merged_output_'
        # # Create the destination folder if it doesn't exist
        if os.path.exists(folder_merged):
            # Remove the folder and its contents
            shutil.rmtree(folder_merged)

        os.makedirs(folder_merged)
        # # Copy images from both folders to the destination images folder
        du.copy_images_source2target(dataset_generated_path, folder_merged)
        du.copy_images_source2target(dataset_train_path, folder_merged)

        # Combine metadata from both folders into single excel sheet
        combined_metadata_path = os.path.join(folder_merged, "metadata.csv")
        du.combine_metadata(dataset_generated_path, dataset_train_path, combined_metadata_path)

        train_loader, test_loader = get_data(folder_merged, dataset_test_path, batch_size=batch_size)
        if verbose:
            print(f'name merged: {folder_merged}')
            print(len(train_loader))
    else:
        train_loader, test_loader = get_data(dataset_train_path, dataset_test_path, batch_size=batch_size)


    print('________________________________________________________________')


    ###########################################################
    # Define model based on the argument parser string.
    print("\n[INFO]: Training the Torchvision ResNet18 model...")
    model = cu.build_model(pretrained=True, fine_tune=True, num_classes=2).to(device)

    if verbose:
        print(model)

    #####################################################
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    ########################################################
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = cu.train(
            model, 
            train_loader, 
            optimizer, 
            criterion, 
            device
        )

        print(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        print("-" * 50)
    
    print("TRAINING COMPLETE")
    print('Initiate testing')
    
    test_loss, test_accuracy, predictions, groundtruth, auc_score = cu.test(
        model, 
        test_loader, 
        criterion, 
        device
        )
    
    print('_________________________________________')
    #print(groundtruth)
    
    print(f'Final values from test: \ntest error:{test_loss:.3f}, \ntest accuracy: {test_accuracy:.3f}, \n AUC score: {auc_score}')