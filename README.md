## MIGROS Budget: Medical Image Generation, Retraining on a Strict Budget

This is the repository for the project for the DL course.
Authors: Luca Anceschi, Marina Crespo Aguirre, Luca Drole

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Computing the FID](#computing-the-fid)
- [Example](#example)

## Introduction

# Training

## Training SeLoRA
The original implementation for the SeLoRA training was made available by Mao et al. 2024 at this [repository](https://anonymous.4open.science/r/SeLoRA-980D). However, this was only used as a base and many changes were made to adapt it to the needs of the project.

To train the SeLoRA make use of the finetuning script: 
```bash
    python selora_finetuning.py
```
while making sure to have a config.yaml set up with the following structure: LOOK AGAIN
```bash
test:
  unet_path: # path where to save the fine tuned unet
  txt_encoder_path: # path where to save the fine tuned text encoder
  imgs_folder:  # path where to save the generated images (I THINK WE DONT USE)

output:
  path: # output path
  folder_name: /selora_outputs
  folder_lora_name: /loras

dataset:
  main_path: # main path to the dataset 
  dataset_name: # specific subfolder
  sequence_type: # modality (i.e. subsubfolder)
  report_name: # name of the report (settled on metadata.csv)
  prompts_for_generation: # full path to a .csv file with same structure as report_name, used to generate the images after training
  

default_random_seed: # random seed used 
batch_size: # batch size
lr: # learning rate
epochs: # number of epochs
th: 10000.0 # lamda value used for the expansion of the LoRA rank

model:
  model_id: runwayml/stable-diffusion-v1-5 # model 
```

After generating the images, some simple post processing (background removal) was done. This was used to correct for issues in background not being homogenous and black (used only for the brain images generated from BraTS). This can be replicated with the postprocess_generated_brats.ipynb notebook.

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

# Computing the FID
The Frechet Inception Distance is a metric used to assess the quality of generated images. To compute the FID you will need a test dataset containing real images and a synthetic dataset. Then, you can run:

```bash
python compute_fid.py --orig /path/to/original --syn /path/to/synthetic
```

# Datasets
For this projects, two publicly available datasets were used.

1. BraTS2021: Brain Tumor Segmentation 2021 Challenge dataset 
Dataset available at [link](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

2. PI-CAI: Prostate Imaging: Cancer AI challenge 
Dataset available at [link](https://zenodo.org/records/6624726)
Labels available at the following [GitHub](https://github.com/DIAGNijmegen/picai_labels/tree/main), for this project only the expert reviewed annotations were used (found under /csPCa_lesion_delineations/human_expert/resampled).

Minor pre-processing has been done with two notebooks (preprocess_data_brats.ipynb and preprocess_data_picai.ipynb). The datasets were processed to extract 2D slices and a metadata.csv document produced that maps every available image with a textual prompt to use for the textual inversion part of the diffusion model. The pre processed datasets are made available for convenience: the ones used for training can be found at FOLDERSNAME, while the testing sets are available under FOLDERSNAME.

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

