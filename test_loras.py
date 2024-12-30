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
from sklearn.model_selection import train_test_split
from IPython.display import display

from config.config_utils import *

def check_and_make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
def clear_cache():
    torch.cuda.empty_cache(); gc.collect(); time.sleep(1); torch.cuda.empty_cache(); gc.collect();



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
unet_file = load_file(config['test']['unet_path'])
txt_encoder_file = load_file(config['test']['txt_encoder_path'])

pipe = StableDiffusionPipeline.from_pretrained(model_id)

print(pipe.text_encoder)


print('----- Replacing modules in model -----')
pipe.text_encoder.load_state_dict(txt_encoder_file)
pipe.unet.load_state_dict(unet_file)


clear_cache()

pipe.to(device)

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
    temp = pipe(temp_prompts, num_inference_steps=30, height = 320, width = 320).images[0]

    temp.save(f'{generate_to_folder}/{place}.png')

    display(temp)