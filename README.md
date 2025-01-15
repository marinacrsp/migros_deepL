# MIGROS Budget: Medical Image Generation, Retraining on a Strict Bufget

This is the repository for the project for the DL course.
Authors: Luca Ansceschi, Marina Crespo Aguirre, Luca Drole

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Computing the FID](#Computing the FID)
- [Example](#example)

## Introduction

# Training

## Training SeLORA

## Training StyleGAN
We use a StyleGAN3 as our baseline. We refer to the [original repository](https://github.com/NVlabs/stylegan3) for further details.
1. To train the StyleGAN, first clone the repository:
    ```bash
    git clone https://github.com/NVlabs/stylegan3.git
    ```
2. Then, you will have to pre-process the data
    ```bash
    python dataset_tool.py --source= /path/to/source --dest= path/to/postprocessed_dataset --resolution=256x256
    ```
3. Now, you can train the StyleGAN. You can regulate the number of iterations with the `kimg` paramter
    ```bash
    python train.py --outdir= /path/to/output_directory --cfg=stylegan3-t --data=path/to/postprocessed_dataset \
        --gpus=1 --batch=3 --gamma=2 --mirror=1 --kimg=25 --snap=6 \
        --resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-ffhqu-256x256.pkl \
        --tick=1 --mbstd-group 1 --metrics none --cbase=16384
    ```
4. Finally, you can perform inference like this:
    ```bash
    python gen_images.py --outdir=/path/to/output_directory --seeds=0-10  --network= /path/to/model
    ```
    Here you can specify any range of seeds to generate the corresponding number of images.
Notice how in this case, one needs to train 2 different models, one for cancer-negative images and one for cancer-positive images.

# Computig the FID
The Frechet Inception Distance is a metric used to assess the quality of generated images. To compute the FID you will need a test dataset containing real images and a 
synthetic dataset. Then, you can run:

```bash
python compute_fid.py --orig /path/to/original --syn /path/to/synthetic
```


# FROM HERE ON IT'S CRAP
## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To compute the FID score, run the script with the required arguments:

```bash
python compute_fid_documented.py --orig /path/to/original --syn /path/to/synthetic
```

### Arguments:
- `--orig`: Path to the folder containing real data (required).
- `--syn`: Path to the folder containing synthetic data (required).
- `--imgsize`: Image size as a tuple (height, width). Default is `(224, 224)`.
- `--mode`: Preprocessing mode. Options: `brats`, `picai`. Default is `brats`.

### Example Command:

```bash
python compute_fid_documented.py --orig ./real_images --syn ./synthetic_images --imgsize 256 256 --mode brats
```

## Inputs and Outputs

### Inputs:
- **Real Images Folder (`--orig`)**: A directory containing real images in `.png`, `.jpg`, or `.jpeg` format.
- **Synthetic Images Folder (`--syn`)**: A directory containing synthetic images in `.png`, `.jpg`, or `.jpeg` format.
- **Image Size (`--imgsize`)**: Tuple specifying the height and width for resizing images. Default: `(224, 224)`.
- **Preprocessing Mode (`--mode`)**: Options for specific dataset preprocessing. Default: `brats`.

### Outputs:
- The script prints the FID score to the console:
  ```
  Original images (real) from: /path/to/original
  Synthetic images (generated) from: /path/to/synthetic
  FID score: 15.73
  ```

## Example

### Example Folder Structure:

```
real_images/
    img1.png
    img2.jpg

synthetic_images/
    img1.png
    img2.jpg
```

### Run Command:

```bash
python compute_fid_documented.py --orig ./real_images --syn ./synthetic_images
```

### Sample Output:

```text
Original images (real) from: ./real_images
Synthetic images (generated) from: ./synthetic_images
FID score: 12.45
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

