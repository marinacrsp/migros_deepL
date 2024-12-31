import os, gc, sys, time, random, math
import torch
from typing import Optional, List
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from sklearn.model_selection import train_test_split
from IPython.display import display
import json
from config.config_utils import *

def check_and_make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
def clear_cache():
    torch.cuda.empty_cache(); gc.collect(); time.sleep(1); torch.cuda.empty_cache(); gc.collect();

def merge_weights(tensor_dict):
    """
    Merges the weights for layers containing weight, lora_A, lora_B according to W' = W + AB^T.

    Args:
        tensor_dict (dict): Dictionary containing the tensor weights.

    Returns:
        dict: Updated tensor dictionary with merged weights.
    """
    merged_tensors = {}

    for key, tensor in tensor_dict.items():
        # Check if this is a weight tensor
        if "weight" in key and "lora_A" not in key and "lora_B" not in key:
            base_key = key.replace(".weight", "")

            lora_A_key = base_key + ".lora_A"
            lora_B_key = base_key + ".lora_B"

            if lora_A_key in tensor_dict and lora_B_key in tensor_dict:
                A = tensor_dict[lora_A_key]
                B = tensor_dict[lora_B_key]
                # Merge weights: W' = W + AB^T
                # (self.lora_B @ self.lora_A)
                rank = A.shape[0]
                scale = 8/rank
                merged_tensor = tensor + scale * (B @ A)

                merged_tensors[base_key + ".weight"] = merged_tensor
            else:
                # Keep the original weight if no lora tensors are found
                merged_tensors[key] = tensor
                
        elif "lora_A" not in key and "lora_B" not in key:
        #     # Copy other tensors as is
            merged_tensors[key] = tensor

    return merged_tensors

WEIGHT_DTYPE = torch.float32

######################################################
############# Load configuration files ###############
args = parse_args()
config = load_config(args.config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


model_id = config["model"]["model_id"]
##############################################################
# Dataset directory & outputs
main_path = config["dataset"]["main_path"]
dataset_name = config["dataset"]["dataset_name"]

## Load the datasets directory and reports path
Data_storage = main_path + dataset_name 
reports_path = Data_storage + config["dataset"]["report_name"]
generate_to_folder = config['test']['imgs_folder']

########################################################
######### Load the pretrained txt encoder / unet ############


print('___________________Testing / Inference phase __________________')
# Load the configurations
with open(config['test']['unet_path'] + 'config.json', 'r') as unet_config_file:
    unet_config = json.load(unet_config_file)

with open(config['test']['txt_encoder_path'] + 'config.json', 'r') as txt_encoder_config_file:
    txt_encoder_config = json.load(txt_encoder_config_file)

# Load the LoRA weights for UNet and text encoder
unet_lora_weights = load_file(config['test']['unet_path'] + 'diffusion_pytorch_model.safetensors')
txt_encoder_lora_weights = load_file(config['test']['txt_encoder_path'] + 'model.safetensors')

print('Merging weights')
unet_lora_merged = merge_weights(unet_lora_weights)
txt_encoder_merged = merge_weights(txt_encoder_lora_weights)


pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
pipe.text_encoder.load_state_dict(txt_encoder_merged)
pipe.unet.load_state_dict(unet_lora_merged)

pipe.to(device)

print('----- Replacing modules in model -----')
clear_cache()

check_and_make_folder(generate_to_folder)

################ Load test data #######################
metadata = pd.read_csv(reports_path)
train_df, temp_df = train_test_split(metadata, test_size=0.2, random_state=32)
valid_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=32)
train_df = train_df[:10] # Generate the first 10 images
################ Generate the new images #################
for place in range(len(train_df)):
    
    temp_prompts = train_df.text.iloc[place]
    print(f'Txt: {temp_prompts}')
    temp = pipe(temp_prompts, num_inference_steps=100, width=256, height=256).images[0]

    temp.save(f'{generate_to_folder}/{place}.png')

    display(temp)