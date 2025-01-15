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
from dataset_utils import *
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
        self.image_transforms = transforms.Compose(
            [

                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

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
    else:
        os.rmdir(path)
        os.mkdir(path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




############################################################
# Learning and training parameters.
# Set seed.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

###############################################################################################
#### Parser arguments
###############################################################################################
parser = argparse.ArgumentParser(description='Computing the AUC')
parser.add_argument('--main', type=str, help='Path to the main folder, brats/ picai', required=True)
parser.add_argument('--test', type=str, help='Path to the folder containing real data for testing', required=True)
parser.add_argument('--train', type=str, help='Path to the folder containing real data for training', required=True)
parser.add_argument('--syn', type=str, help='Path to the folder containing synthetic data to be appended to the train', required=True)
parser.add_argument('--merge', type=str, help='Specify whether to augment training data(1) or not(0)', required=True)
parser.add_argument('--epochs', type=str, help='N# of training epochs', required=False)
parser.add_argument('--batch_size', type=str, help='Size of training batch', required=False)
parser.add_argument('--lr', type=str, help='Learning rate', required=False)
args = parser.parse_args()


main_path = args.main
dataset_train_path = main_path + args.train
dataset_test_path = main_path + args.test
dataset_generated_path = main_path + args.syn
merge = args.merge


if args.epochs == None:
    epochs= 50
else:
    epochs = int(args.epochs)

if args.batch_size == None:
    batch_size = 32
else:
    batch_size = int(args.batch_size)

if args.lr == None:
    learning_rate = 1.e-4
else:
    learning_rate = float(args.lr)


print(f'Inputted Variables: synthetic:{dataset_generated_path}, test:{dataset_test_path}, train:{dataset_train_path}, \n merge: {merge} ')

if merge == 'True':
    folder_merged = main_path + 'merged_output_' + args.syn
    # Replace with the path to the destination folder TODO
    # # Create the destination folder if it doesn't exist
    if os.path.exists(folder_merged):
        # Remove the folder and its contents
        shutil.rmtree(folder_merged)

    os.makedirs(folder_merged)
    # # Copy images from both folders to the destination images folder
    copy_images_source2target(dataset_generated_path, folder_merged)
    copy_images_source2target(dataset_train_path, folder_merged)

    # Combine metadata from both folders into single excel sheet
    combined_metadata_path = os.path.join(folder_merged, "metadata.csv")
    combine_metadata(dataset_generated_path, dataset_train_path, combined_metadata_path)


    train_loader, test_loader = get_data(folder_merged, dataset_test_path, batch_size=batch_size)
    print(f'name merged: {folder_merged}')
    print(len(train_loader))
elif merge == 'False':
    train_loader, test_loader = get_data(dataset_train_path, dataset_test_path, batch_size=batch_size)



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
    
    test_loss, test_accuracy, predictions, groundtruth, auc_score = test(
        model, 
        test_loader, 
        criterion, 
        device
        )
    
    print('_________________________________________')
    print(groundtruth)
    
    print(f'Final values from test: \ntest error:{test_loss:.3f}, \ntest accuracy: {test_accuracy:.3f}, \n AUC score: {auc_score}')