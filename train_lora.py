# -*- coding: utf-8 -*-

import os, gc, sys, time, random, math

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

from PIL import Image

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config_utils import *
from selora_functions import *


############# Load configuration files #################
args = parse_args()
config = load_config(args.config)


##### Setting Random Seed
DEFAULT_RANDOM_SEED = config["default_random_seed"]
WEIGHT_DTYPE = torch.float32
BATCH_SIZE = config["batch_size"]
LR = config["lr"]
##############################################################
# Dataset directory & outputs
main_path = config["dataset"]["main_path"]
dataset_name = config["dataset"]["dataset_name"]

## Load the datasets directory and reports path
Data_storage = main_path + dataset_name 
reports_path = Data_storage + config["dataset"]["report_name"]

## Load the output directory and folder_name, where the results will be printed
output_path = config["output"]["path"] + '/' + config["timestamp"]
output_folder = config["output"]["folder_name"]
folder_name = config["output"]["folder_lora_name"]

save_result_path = output_path + output_folder


print(f'{main_path}, \n {output_path}, \n {save_result_path}')
#########################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = config["model"]["model_id"]
set_rank = config['model']['rank']
UNET_TARGET_MODULES = list(config["model"]["unet_modules"]) # <- marina code
TEXT_ENCODER_TARGET_MODULES = list(config["model"]["txt_encoder_modules"]) # <- marina code

# UNET_TARGET_MODULES = [
#     "to_q", "to_k", "to_v",
#     "proj", "proj_in", "proj_out",
#     "conv", "conv1", "conv2",
#     "conv_shortcut", "to_out.0", "time_emb_proj", "ff.net.2",
# ]

# TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]


def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seedBasic(); seedTorch()

def clear_cache():
    torch.cuda.empty_cache(); gc.collect(); time.sleep(1); torch.cuda.empty_cache(); gc.collect();


def check_and_make_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)

def print_trainable_parameters(model):

    total_parameter_count = sum([np.prod(p.size()) for p in model.parameters()]) # Computes the total number of layers/ parameters in the layer (weight and bias)
    for name, layer in model.named_modules():
        # Freeze the temporal expanding matrixes lora_A and lora_B - for trainable parameter computation
        # They are active at times during the computational graph
        if isinstance(layer, Linear):
            layer.lora_A_temp.requires_grad = False
            layer.lora_B_temp.requires_grad = False
            total_parameter_count -= np.prod(layer.lora_A_temp.size()) + np.prod(layer.lora_B_temp.size())

    trainable_parameter = filter(lambda p: p.requires_grad, model.parameters()) # Filters the parameters out of all the parameters that require gradient
    trainable_parameter_count = sum([np.prod(p.size()) for p in trainable_parameter]) # Gives the sum of parameters that require gradient, p is the matrix of parameters, product of columns and rows = total num of parameters
    trainable_percentage = (trainable_parameter_count / total_parameter_count)  * 100

    formatted_output = (
        f"trainable params: {trainable_parameter_count:,} || "
        f"all params: {total_parameter_count:,} || "
        f"trainable%: {trainable_percentage:.16f}"
    )

    ## Unfreeze it for training purposes
    for name, layer in model.named_modules():
        if isinstance(layer, Linear):
            layer.lora_A_temp.requires_grad = True
            layer.lora_B_temp.requires_grad = True

    print(formatted_output)


def remove_param_from_optimizer(optim, param):
    for j in range(len(optim.param_groups)):
        optim_param_group_list = optim.param_groups[j]["params"]
        for i, optim_param in enumerate(optim_param_group_list):
            if param.shape == optim_param.shape and (param==optim_param).all():
                del optim.param_groups[j]["params"][i]


"""# Hyperparameters and Path"""
# assert not os.path.exists(f'{save_result_path}/{folder_name}'), print('LoRA Experiment Already Run')
check_and_make_folder(f'{save_result_path}')
check_and_make_folder(f'{save_result_path}/{folder_name}')

pipe = StableDiffusionPipeline.from_pretrained(model_id)
tokenizer = pipe.tokenizer
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

# Freeze the Bulk part of the model
text_encoder = text_encoder.requires_grad_(False)
vae =vae.requires_grad_(False)
unet = unet.requires_grad_(False)

unet_lora = set_Linear_SeLoRA(unet, UNET_TARGET_MODULES, set_rank) # removed: set_rank
text_encoder_lora = set_Linear_SeLoRA(text_encoder, TEXT_ENCODER_TARGET_MODULES, set_rank) # removed: set rank


### Print the parameters in the Unet and Text encoder that will be trained
print_trainable_parameters(text_encoder_lora)
print_trainable_parameters(unet_lora)


metadata = pd.read_csv(reports_path)
# Split the data into train, validation, and test sets
train_df, temp_df = train_test_split(metadata, test_size=0.2, random_state=DEFAULT_RANDOM_SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.2, random_state=DEFAULT_RANDOM_SEED)

# ImageDataset initialization
train_ds = ImageDataset(
    root_dir=Data_storage,
    df=train_df,
    tokenizer=tokenizer,
)

# valid_ds = ImageDataset(
#     root_dir=Data_storage,
#     df=valid_df,
#     tokenizer=tokenizer,
# )

test_ds = ImageDataset(
    root_dir=Data_storage,
    df=test_df,
    tokenizer=tokenizer,
)
# print(f'n of working cpus: {os.cpu_count()}')
num_workers = 2
# DataLoader initialization
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = num_workers)
# valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers = num_workers)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers = num_workers)
print(f'Training set of size: {len(train_df)}, and real train set of size: {len(train_ds)}')
optimizer = torch.optim.Adam(list(unet_lora.parameters()) + list(text_encoder_lora.parameters()), lr=LR)

###############################################################################
"""# Training"""

from tqdm.notebook import tqdm
from IPython.display import display
import copy


class Trainer:
    def __init__(self, vae, unet, text_encoder, noise_scheduler, optimizer, train_dl, test_dl, total_epoch, WEIGHT_DTYPE, threshould = 2, per_iter_valid = 60, log_period = 20, expand_step = 20):
        self.vae = vae.to(device, dtype=WEIGHT_DTYPE)
        self.unet = unet.to(device, dtype=WEIGHT_DTYPE)
        self.text_encoder = text_encoder.to(device, dtype=WEIGHT_DTYPE)
        self.noise_scheduler = noise_scheduler
        self.optimizer = optimizer
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.WEIGHT_DTYPE = WEIGHT_DTYPE
        self.total_epoch = total_epoch
        self.threshould = threshould
        self.per_iter_valid = per_iter_valid
        self.total_step = 0
        self.result_df = pd.DataFrame(columns=['epoch', 'steps', 'Train Loss', 'Valid Loss', 'Total Added Rank', 'unet trainable', 'text_encoder trainable'])
        self._display_id = None
        self.log_period = log_period
        self.expand_step = expand_step

        self.best_text_encoder = None
        self.best_unet = None
        self.display_line = ''
        self.added_rank = 1

        print(f'total steps: {len(train_dl) * total_epoch}')

    def Expandable_LoRA(self, model):
        
        for name, layer in model.named_modules():
            if isinstance(layer, Linear):
                self.display_line += f'{layer.get_ratio():.4f}, {layer.get_active_rank()}   '
                if layer.get_ratio() >= self.threshould:
                    self.added_rank += 1
                    self.optimizer = layer.expand_rank(self.optimizer)
                    
        print(self.rank_display_id)
        # self.rank_display_id.update(self.display_line)


    def valid(self):
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()

        valid_pbar = tqdm(self.test_dl, desc = 'validating', leave = False)

        valid_loss, number_of_instance = [], 0

        for step, batch in enumerate(valid_pbar):

            pixel_values = batch["instance_images"].to(device, dtype=self.WEIGHT_DTYPE)
            pormpt_idxs  = batch["instance_prompt_ids"].to(device).squeeze(1)

            # Convert images to latent space
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents) # Randomly generated gaussian - groundtruth epsilon
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(pormpt_idxs)[0]
            # Predict the noise residual (x0 - xt)
            model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
            target = noise

            ## MSE - epsilon_t vs pred_epsilon_t
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            valid_loss.append(loss.item() * len(batch))
            number_of_instance += len(batch)
            
            clear_cache()
            # torch.cuda.empty_cache()

        ########################################################################
        ## add log and save model here TODO
        ########################################################################

        self.unet.train()
        self.vae.train()
        self.text_encoder.train()

        torch.cuda.empty_cache()

        return sum(valid_loss) / number_of_instance

    def trainable_percentage(self, model):
        total_parameter_count = sum([np.prod(p.size()) for p in model.parameters()])
        trainable_parameter_count = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
        return (trainable_parameter_count / total_parameter_count)  * 100

    def model_to_temp(self, model):
        for name, layer in model.named_modules():
            if isinstance(layer, Linear):
                layer.change_to_temp()



    def train(self):

        self._display_id = display(self.result_df, display_id=True)
        self.rank_display_id = display('', display_id=True)
        
        self.vae.train()
        self.unet.train()
        self.text_encoder.train()

        recorded_loss = []

        for epoch in range(self.total_epoch):

            pbar = tqdm(self.train_dl) # The training bar ????
            for step, batch in enumerate(pbar):

                pixel_values = batch["instance_images"].to(device, dtype=self.WEIGHT_DTYPE)
                pormpt_idxs  = batch["instance_prompt_ids"].to(device).squeeze(1)

                # Convert images to latent space
                latents = self.vae.encode(pixel_values).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = self.text_encoder(pormpt_idxs)[0]
                # Predict the noise residual
                model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                self.optimizer.zero_grad()

                loss.backward()

                recorded_loss.append(loss.item())

                pbar.set_description(f"[Loss: {recorded_loss[-1]:.3f}/{np.mean(recorded_loss):.3f}]")

                self.optimizer.step()
                print(f'Training error at step {step}: {loss.item()}')
                self.total_step += 1
                
                # #######################################################################
                # #### Commenting the expandable lora's 
                # if self.total_step % self.expand_step == 0:
                #     self.model_to_temp(self.unet)
                #     self.model_to_temp(self.text_encoder)

                #     # Get the text embedding for conditioning
                #     encoder_hidden_states = self.text_encoder(pormpt_idxs)[0]
                #     # Predict the noise residual
                #     model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                #     loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                #     self.optimizer.zero_grad()

                #     loss.backward()


                #     self.display_line = ''
                #     self.Expandable_LoRA(self.unet)
                #     self.Expandable_LoRA(self.text_encoder)


                #     self.model_to_temp(self.unet)
                #     self.model_to_temp(self.text_encoder)


                ######################################################################

                clear_cache()
                # if self.total_step % self.per_iter_valid == 0:

                #     valid_rmse = self.valid()

                #     if valid_rmse <= min([x for x in trainer.result_df['Valid Loss'] if x != ' --- '] + [1.0]):

                #         self.best_text_encoder = copy.deepcopy(self.text_encoder).cpu()
                #         self.best_unet = copy.deepcopy(self.unet).cpu()

                #         check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model')
                #         check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
                #         check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Text')
                #         print(f'Best validation error so far in epoch {epoch}, step {step}: {valid_rmse}')
                #         self.unet.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
                #         self.text_encoder.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Text')


                #     self.result_df.loc[len(self.result_df)] = [epoch, self.total_step, np.round(np.mean(recorded_loss), 4), np.round(valid_rmse, 4), self.added_rank,  self.trainable_percentage(self.unet), self.trainable_percentage(self.text_encoder)]

                #     print(self.result_df)
                #     recorded_loss = []

                if self.total_step % self.log_period == 0:
                    self.result_df.loc[len(self.result_df)] = [epoch, self.total_step, np.round(np.mean(recorded_loss), 4), ' --- ', self.added_rank,  self.trainable_percentage(self.unet), self.trainable_percentage(self.text_encoder)]
                    self.result_df.to_csv(f'{save_result_path}/{folder_name}/results.csv')

        print('Saving model')
        self.best_text_encoder = copy.deepcopy(self.text_encoder).cpu()
        self.best_unet = copy.deepcopy(self.unet).cpu()

        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model')
        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Text')
        self.unet.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
        self.text_encoder.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Text')



trainer = Trainer(
    vae = vae,
    unet = unet_lora,
    text_encoder = text_encoder_lora,
    noise_scheduler = noise_scheduler,
    optimizer = optimizer,
    train_dl = train_loader,
    test_dl = test_loader,
    total_epoch = 10, #10,
    WEIGHT_DTYPE = WEIGHT_DTYPE,
    threshould = 11.3,
    per_iter_valid = len(train_loader),
    log_period = 40,
    expand_step = 40,
)


trainer.train()

trainer.result_df.to_csv(f'{save_result_path}/{folder_name}/results.csv')



"""# Testing / Inference"""
print('___________________Testing / Inference phase __________________')

unet_lora.eval()
text_encoder_lora.eval()
clear_cache()
new_pipe = StableDiffusionPipeline(
    tokenizer=tokenizer,
    text_encoder=trainer.best_text_encoder.to(device, dtype=WEIGHT_DTYPE),
    vae=vae,
    unet=trainer.best_unet.to(device, dtype=WEIGHT_DTYPE),
    scheduler=noise_scheduler,
    safety_checker= None,
    feature_extractor=None
)

new_pipe.to(device)

check_and_make_folder(f'{save_result_path}/{folder_name}')
check_and_make_folder(f'{save_result_path}/{folder_name}/test_images')

## Visualize the first 10 prompts
if len(test_df)>10:
    test_df = test_df[:10]

for place in range(len(test_df)):
    temp_prompts = test_df.text.iloc[place]
    temp = new_pipe(temp_prompts, height = 224, width = 224).images[0]
    temp.save(f'{save_result_path}/{folder_name}/test_images/{place}.png')
    display(temp)