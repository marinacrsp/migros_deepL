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
    Adds 2 channels to a grayscale image. Needed for FID calculation.
    """
    return input_tensor.repeat(1, 3, 1, 1)


class ImageDataset(Dataset):
    def __init__(self, folder_path, mode='brats', synthetic=False):
        self.folder_path = folder_path
        self.mode = mode
        self.synthetic = synthetic
        self.images = [
            f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpeg'))
        ]
        if not self.images:
            raise ValueError(f"No images found in the folder '{folder_path}'.")

        self.transform = self.get_transform()

    def get_transform(self):
        if self.mode == 'brats' or self.mode == 'picai':
            size = 224
            center_crop = True
            return transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                    transforms.ToTensor(),
                ]
            )
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Select a valid origin.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)


def main():
  parser = argparse.ArgumentParser(description='Computing the FID')
  parser.add_argument('--orig', type=str, help='Path to the folder containing real data', required=True)
  parser.add_argument('--syn', type=str, help='Path to the folder containing synthetic data', required=True)
  parser.add_argument('--imgsize', type=tuple, help='Image size as a tuple (N, M). Default (224,224)', default=DEFAULT_IMAGE_SIZE, nargs='+')
  parser.add_argument('--mode', type=str, help='Modality associated to different datasets. Default: brats', default='brats', nargs='+')
  args = parser.parse_args()
  folder_original = args.orig
  folder_sythetic = args.syn
  img_size = args.imgsize
  mode = args.mode

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare datasets and loaders
  original_dataset = ImageDataset(folder_original, mode=mode, synthetic=False)
  synthetic_dataset = ImageDataset(folder_sythetic, mode=mode, synthetic=True)

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
  print(f'Synthetic images (generated) from: {folder_sythetic}')
  print(f'FID score: {fid_result.item()}')


if __name__ == "__main__":
    main()
