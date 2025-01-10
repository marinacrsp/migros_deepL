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
from classifier_model import build_model
from classifier_utils import *

import sys
import os

# Get the absolute path to the 'config' folder
big_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "/.."))
config_path = os.path.join(big_folder_path, "config", "config_utils.py")
# Add the 'config' folder to the Python module search path
sys.path.append(config_path)
from config.config_utils import *




class ImageDataset(Dataset):
    def __init__(self, root_dir, df, size = 224, center_crop = True, train=True):
        self.root_dir = root_dir
        self.files = df['file_name'].tolist()
        self.labels = df['text'].tolist()

        if train:
            self.image_transforms = transforms.Compose(
                [
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=1, contrast=0, saturation=0, hue=0)
                ]
            )
        else: #test data
            print('No training transformations added')
            self.image_transforms = transforms.Compose(
                [

                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )

    def add_gaussian_noise(img, mean=0.0, std=1.0):
        if isinstance(img, torch.Tensor):
            noise = torch.randn(img.size()) * std + mean
            return img + noise
        else:
            raise TypeError("Input must be a PyTorch tensor.")

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
    metadata_train = pd.read_csv(data_train + 'metadata.csv')
    metadata_test = pd.read_csv(data_test + 'metadata.csv')
    print(f'\nTrain set: {len(metadata_train)}, \nTest set: {len(metadata_test)}')

    dataset_train = ImageDataset(
    root_dir=data_train,
    df=metadata_train,
    train =False,
)
    dataset_test = ImageDataset(
    root_dir=data_test,
    df=metadata_test,
    train=False,
    )

    # Create data loaders.
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False) 
    return train_loader, test_loader


def check_and_make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = parse_args()
config = load_config(args.config)

############################################################
# Learning and training parameters.
# Set seed.
seed = config['classifier']['seed']
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

epochs = config["classifier"]["training"]["n_epochs"]
batch_size = config["classifier"]["training"]["batch_size"]
learning_rate = config["classifier"]["training"]["lr"]

dataset_train = config['classifier']['dataset']['data_train']
dataset_test = config['classifier']['dataset']['data_test']

train_loader, test_loader = get_data(dataset_train, dataset_test, batch_size=batch_size)
print('________________________________________________________________')

###########################################################
# Define model based on the argument parser string.
print("\n[INFO]: Training the Torchvision ResNet18 model...")
model = build_model(pretrained=True, fine_tune=True, num_classes=2).to(device)

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


if __name__ == "__main__":
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
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
    
    # save_plots(train_acc, valid_acc, train_loss, valid_loss, plot_folder)
    print("TRAINING COMPLETE")
    print('Initiate testing')
    # Save the loss and accuracy plots.
    
    test_loss, test_accuracy, predictions, groundtruth = test(
        model, 
        test_loader, 
        criterion, 
        device
        )
    
    print('_________________________________________')
    print(groundtruth)
    
    print(f'Final values from test: \ntest error:{test_loss:.3f}, \ntest accuracy: {test_accuracy:.3f}')