import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
from torchmetrics.image.fid import FrechetInceptionDistance


FOLDER_ORIGINAL = '/home/ldrole/migros_deepL/selected_from_brats_1000/Flair'
FOLDER_SYNTHETIC = '/work/scratch/ldrole/output/test_26_12/12-31_20h21m54s/selora_outputs/loras/test_images'
MODE = 'brats' 
IMAGE_SIZE = (224,224)

def make3channel(input_tensor):
    '''
    Adds 2 channels to a greyscale image. Needed for FID calculation.
    '''
    # Get the dimensions of the input tensor
    batch_size, height, width = input_tensor.size()

    # Reshape the input tensor to have a new dimension for channels
    input_tensor = input_tensor.unsqueeze(1)

    # Replicate each sample in the batch three times
    input_tensor = input_tensor.repeat(1, 3, 1, 1)

    return input_tensor

def process_images(folder_path, mode = 'brats', synthetic=False):
    """
    Checks if a folder is empty, and if not, processes all images 
    in the folder, converting them to grayscale and storing them in a NumPy array. Furthermore,
    ensures that the images in the original dataset are consistent with the generated ones.

    Args:
        folder_path: The path to the folder.
        mode: the origin of the data the function is being used on.
        synthetic: If the images are synthetic they do not need to be resized
    Returns:
        A NumPy array containing the grayscale images, or raises an error if 
        the folder is empty or if there are issues with image processing.

    Raises:
        ValueError: If the folder is empty or if no PNG images are found.
        FileNotFoundError: If the folder does not exist
        IOError: If there is a problem while reading the image files.
    """
    if not os.path.exists(folder_path):
      raise FileNotFoundError(f"Error: The folder path '{folder_path}' does not exist.")

    if not os.listdir(folder_path):
        raise ValueError(f"Error: The folder '{folder_path}' is empty.")

    images = [f for f in os.listdir(folder_path) if f.lower().endswith('.png') or f.lower().endswith('.jpeg')]
    
    if not images:
        raise ValueError(f"Error: No images found in the folder '{folder_path}'.")

    image_array = []
    for image_file in images:
        try:
          image_path = os.path.join(folder_path, image_file)
          img = Image.open(image_path).convert('L')  # Open as grayscale
          image_array.append(np.array(img))
        except IOError:
          raise IOError(f"Error: Unable to open or process image file '{image_file}'.")

    # Make sure the images are consistent between synthetic and real
    if mode == 'brats':
      size = 224
      center_crop = True
      if not synthetic:
        brats_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            ]
        )
        image_array = [brats_transforms(Image.fromarray(img)) for img in image_array]
    else:
        raise ValueError(f'Unknown mode {mode}. Select a valid origin')
    
    return np.array(image_array)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_images = process_images(FOLDER_ORIGINAL)
    synthetic_images = process_images(FOLDER_SYNTHETIC, synthetic=True)
    
    # Compute FID
    fid = FrechetInceptionDistance(normalize=True, input_img_size=(3,IMAGE_SIZE[0], IMAGE_SIZE[1])).to(device)
    fid.update(make3channel(torch.tensor(synthetic_images).to(device)), real=False)
    fid.update(make3channel(torch.tensor(original_images).to(device)), real=True)
    fid_result = fid.compute()
    
    # Output
    print(f'Original images (real) from: {FOLDER_ORIGINAL}')
    print(f'Sythetic images (generated) from: {FOLDER_SYNTHETIC}')
    print(f'FID score: {fid_result}')
    
if __name__ == "__main__":
    main()
