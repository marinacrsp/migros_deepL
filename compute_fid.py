import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader, Dataset
import argparse

DEFAULT_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def make3channel(input_tensor):
    """
    Adds 2 additional channels to a grayscale image to convert it to a 3-channel format.
    This is required for the FID calculation which expects 3-channel input images.
    """
    return input_tensor.repeat(1, 3, 1, 1)


class ImageDataset(Dataset):
    """
    Custom dataset for loading and preprocessing images from a folder.
    
    Inputs:
        folder_path (str): Path to the folder containing the images.
        
        mode (str): Specifies the dataset mode ('brats' or 'picai'). Default is 'brats'. As of now, no difference.
        In the future, this could be used to apply different transformations based on the dataset.
        
        synthetic (bool): Indicates whether the dataset is synthetic or not. Default is False. As of now, no difference.
        In the future, this could be used to apply different transformations for synthetic data.

    """
    def __init__(self, folder_path, mode='brats', synthetic=False):
        self.folder_path = folder_path
        self.mode = mode
        self.synthetic = synthetic

        # Get all valid image files in the folder
        self.images = [
            f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpeg', '.jpg'))
        ]
        if not self.images:
            raise ValueError(f"No valid images found in the folder '{folder_path}'.")

        self.transform = self.get_transform()

    def get_transform(self):
        """
        Returns the transformation pipeline for preprocessing images.
        
        Returns:
          torchvision.transforms.Compose: The transformation pipeline.
        """
        if self.mode in ['brats', 'picai']:
            size = 224
            center_crop = True
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Valid options are 'brats' or 'picai'.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Loads and transforms an image by index.
        
        Args:
          idx (int): Index of the image to load.
        Returns:
          torch.Tensor: The transformed image tensor.
        """
        image_path = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


def validate_folder(path, folder_type):
    """
    Validates that the given folder path exists and contains valid images.
    """
    if not os.path.isdir(path):
        raise ValueError(f"{folder_type} folder '{path}' does not exist or is not accessible.")
    if not any(file.lower().endswith(('.png', '.jpeg', '.jpg')) for file in os.listdir(path)):
        raise ValueError(f"No valid image files found in the {folder_type} folder '{path}'.")


def main():
  parser = argparse.ArgumentParser(description='Compute the Frechet Inception Distance (FID) score.')
  parser.add_argument('--orig', type=str, required=True, help='Path to the folder containing real data.')
  parser.add_argument('--syn', type=str, required=True, help='Path to the folder containing synthetic data.')
  parser.add_argument('--imgsize', type=int, nargs=2, default=DEFAULT_IMAGE_SIZE,
                      help='Image size as a tuple (height, width). Default is (224, 224).')
  parser.add_argument('--mode', type=str, default='brats',
                      help="Dataset mode. Options: 'brats', 'picai'. Default is 'brats'.")

  args = parser.parse_args()
  folder_original = args.orig
  folder_synthetic = args.syn
  img_size = tuple(args.imgsize)
  mode = args.mode
  # Validate inputs
  validate_folder(folder_original, "Original data")
  validate_folder(folder_synthetic, "Synthetic data")

  # Set device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Prepare datasets and loaders
  original_dataset = ImageDataset(folder_original, mode=mode, synthetic=False)
  synthetic_dataset = ImageDataset(folder_synthetic, mode=mode, synthetic=True)

  original_loader = DataLoader(original_dataset, batch_size=BATCH_SIZE, shuffle=False)
  synthetic_loader = DataLoader(synthetic_dataset, batch_size=BATCH_SIZE, shuffle=False)

  # Initialize FID metric
  fid = FrechetInceptionDistance(normalize=True, input_img_size=(3, img_size[0], img_size[1])).to(device)

  # Update FID incrementally with batched inputs
  for original_batch in original_loader:
    fid.update(make3channel(original_batch.to(device)), real=True)


  for synthetic_batch in synthetic_loader:
      fid.update(make3channel(synthetic_batch.to(device)), real=False)


    # Compute FID score
  fid_result = fid.compute()

    # Output
  print(f'Original images (real) from: {folder_original}')
  print(f'Synthetic images (generated) from: {folder_synthetic}')
  print(f'FID score: {fid_result.item()}')


if __name__ == "__main__":
    main()
