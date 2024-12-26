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
modality = config["dataset"]["sequence_type"]

## Load the datasets directory and reports path
Data_storage = main_path + dataset_name + modality
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
UNET_TARGET_MODULES = list(config["model"]["unet_modules"])
TEXT_ENCODER_TARGET_MODULES = list(config["model"]["txt_encoder_modules"])


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

seedBasic(); seedTorch();

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

clear_cache()

"""# Hyperparameters and Path"""



# assert not os.path.exists(f'{save_result_path}/{folder_name}'), print('LoRA Experiment Already Run')
check_and_make_folder(f'{save_result_path}')
check_and_make_folder(f'{save_result_path}/{folder_name}')



# UNET_TARGET_MODULES = [
#     "to_q", "to_k", "to_v",
#     "proj", "proj_in", "proj_out",
#     "conv", "conv1", "conv2",
#     "conv_shortcut", "to_out.0", "time_emb_proj", "ff.net.2",
# ]


# TEXT_ENCODER_TARGET_MODULES = ["fc1", "fc2", "q_proj", "k_proj", "v_proj", "out_proj"]


pipe = StableDiffusionPipeline.from_pretrained(model_id)
tokenizer = pipe.tokenizer
noise_scheduler = pipe.scheduler
text_encoder = pipe.text_encoder
vae = pipe.vae
unet = pipe.unet

# Freeze the Bulk part of the model
text_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
print()

def remove_param_from_optimizer(optim, param):
    for j in range(len(optim.param_groups)):
        optim_param_group_list = optim.param_groups[j]["params"]
        for i, optim_param in enumerate(optim_param_group_list):
            if param.shape == optim_param.shape and (param==optim_param).all():
                del optim.param_groups[j]["params"][i]

clear_cache()

"""# SeLoRA class
Below Defines the SeLoRA injected linear layer class, shown in `Linear`.
"""

class LoRALayer():
    """Function that introduces functionality to augment the neural network layers with LoRA parameters
    arguments:
    - r: rank
    - lora_alpha: scaling factor that adjusts the influence of lora matrixes on the weights
    - lora_dropout: dropout rate for lora specific parameter
    - merge_weights: determines whether Lora weights should be merged back into original matrixes Wo
    """
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

### This is the SeLoRA replacement of nn.Linear layer
class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 8,
        lora_dropout: float = 0.,
        EMA_factor: float = 0.6,
        # warmup_step_per_expand:int = 10,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.r = r
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

            self.lora_A_temp = nn.Parameter(self.weight.new_zeros((r + 1, in_features)))
            self.lora_B_temp = nn.Parameter(self.weight.new_zeros((out_features, r + 1)))

            self.use_temp_weight = False

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False

            self.recorded_grad = 1

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def get_active_rank(self):
        assert self.lora_A.shape[0] == self.lora_B.shape[1]
        return self.lora_A.shape[0]

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.zeros_(self.lora_B_temp)
            nn.init.kaiming_uniform_(self.lora_A)


    def train(self, mode: bool = True):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
            self.merged = False

    def eval(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:

                self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling

            self.merged = True

    def forward(self, x: torch.Tensor,*args,  **kwargs):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)

            if self.r > 0:
                if not self.use_temp_weight:
                    result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
                else:
                    self.lora_A_temp.data[:-1, :] = self.lora_A.data
                    self.lora_B_temp.data[:, :-1] = self.lora_B.data

                    result += (self.lora_dropout(x) @ self.lora_A_temp.T @ self.lora_B_temp.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def sum_square(self, _A, _B):
        return torch.sum(torch.square(_A)) + torch.sum(torch.square(_B))


    def get_rmse_grad(self):
        if not self.use_temp_weight:
            return self.sum_square(self.lora_A.grad, self.lora_B.grad)
        else:
            return self.sum_square(self.lora_A_temp.grad, self.lora_B_temp.grad)

    def change_to_temp(self):
        if not self.use_temp_weight:
            self.recorded_grad = self.get_rmse_grad()
            self.use_temp_weight = True
            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False
            self.lora_A_temp.requires_grad = True
            self.lora_B_temp.requires_grad = True
        else:
            self.use_temp_weight = False
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True
            self.lora_A_temp.requires_grad = False
            self.lora_B_temp.requires_grad = False


    def get_ratio(self):
        if not self.use_temp_weight:
            return 0
        return self.get_rmse_grad() / self.recorded_grad



    def expand_rank(self, optimizer):

        old_lora_A = self.lora_A.data
        remove_param_from_optimizer(optimizer, self.lora_A)
        self.lora_A = nn.Parameter(self.weight.new_zeros((self.lora_A.shape[0] + 1, self.lora_A.shape[1])))
        nn.init.kaiming_uniform_(self.lora_A)
        self.lora_A.data[:-1, :] = old_lora_A


        old_lora_B = self.lora_B.data
        remove_param_from_optimizer(optimizer, self.lora_B)
        self.lora_B = nn.Parameter(self.weight.new_zeros((self.lora_B.shape[0], self.lora_B.shape[1] + 1)))
        nn.init.zeros_(self.lora_B)
        self.lora_B.data[:, :-1] = old_lora_B

        remove_param_from_optimizer(optimizer, self.lora_A_temp)
        remove_param_from_optimizer(optimizer, self.lora_B_temp)
        self.lora_A_temp = nn.Parameter(self.weight.new_zeros((self.lora_A.shape[0] + 1, self.lora_A.shape[1])))
        nn.init.kaiming_uniform_(self.lora_A_temp)
        self.lora_B_temp = nn.Parameter(self.weight.new_zeros((self.lora_B.shape[0], self.lora_B.shape[1] + 1)))
        nn.init.zeros_(self.lora_B_temp)

        optimizer.add_param_group({'params': self.lora_A})
        optimizer.add_param_group({'params': self.lora_B})
        optimizer.add_param_group({'params': self.lora_A_temp})
        optimizer.add_param_group({'params': self.lora_B_temp})

        return optimizer


text_encoder = pipe.text_encoder
text_encoder = text_encoder.requires_grad_(False)
unet = pipe.unet
unet = unet.requires_grad_(False)

def set_Linear_SeLoRA(model, target_modules, rank):
     # replace all linear layer (include q,k,v layer) into DyLoRA Layer.

    for name, layer in model.named_modules():
        
        if isinstance(layer, nn.Linear):

            if name.split('.')[-1] in target_modules: # NOTE: Check if the name of current layer is contained in the target modules

                print(f"Replacing layer {name.split('.')[-1]} in {layer}")
                LoRA_layer = Linear(
                    in_features = layer.in_features,
                    out_features = layer.out_features,
                    r = rank
                )
                LoRA_layer.weight = layer.weight
                LoRA_layer.weight.requires_grad = True
                LoRA_layer.bias = layer.bias

                if LoRA_layer.bias != None:
                    LoRA_layer.bias.requires_grad = True

                pointing_layer = model

                if name.split('.')[-1] in target_modules: 

                    for layer_name in name.split('.')[:-1]:
                        pointing_layer = getattr(pointing_layer, layer_name)

    return model

# def set_Linear_SeLoRA(model, target_modules):
#      # replace all linear layer (include q,k,v layer) into DyLoRA Layer.
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Linear):

#             LoRA_layer = Linear(
#                   in_features = layer.in_features,
#                   out_features = layer.out_features,
#                   r = 1
#             )

#             LoRA_layer.weight = layer.weight
#             LoRA_layer.weight.requires_grad = False
#             LoRA_layer.bias = layer.bias
#             if LoRA_layer.bias != None:
#                 LoRA_layer.bias.requires_grad = False

#             pointing_layer = model
#             if len(target_modules) != len(None):
#                 if name.split('.')[-1] in target_modules:
#                     for layer_name in name.split('.')[:-1]:
#                         pointing_layer = getattr(pointing_layer, layer_name)
#             else:
#                 for layer_name in name.split('.')[:-1]:
#                         pointing_layer = getattr(pointing_layer, layer_name)

#                 setattr(pointing_layer, name.split('.')[-1], LoRA_layer)
#     return model

# def set_Linear_SeLoRA(model, target_modules):
#     # Replace all linear layers into LoRA layers
#     for name, layer in model.named_modules():
#         if isinstance(layer, nn.Linear):
#             print(f"Checking layer {name} for inclusion in target_modules...")
#             if len(target_modules) != 0 or name.split('.')[-1] in target_modules:
#                 print(f"Layer {name} added to LoRA replacement")

#                 LoRA_layer = Linear(
#                     in_features=layer.in_features,
#                     out_features=layer.out_features,
#                     r=1
#                 )
#                 LoRA_layer.weight = layer.weight
#                 LoRA_layer.weight.requires_grad = True
#                 LoRA_layer.bias = layer.bias
#                 if LoRA_layer.bias is not None:
#                     LoRA_layer.bias.requires_grad = True

#                 pointing_layer = model
#                 for layer_name in name.split('.')[:-1]:
#                     pointing_layer = getattr(pointing_layer, layer_name)

#                 # Replace the layer
#                 setattr(pointing_layer, name.split('.')[-1], LoRA_layer)
#                 print(f"Replaced {name} with LoRA layer")
#     return model

unet_lora = set_Linear_SeLoRA(unet, UNET_TARGET_MODULES, set_rank)
text_encoder_lora = set_Linear_SeLoRA(text_encoder, TEXT_ENCODER_TARGET_MODULES, set_rank)


### Print the parameters in the Unet and Text encoder that will be trained
print_trainable_parameters(text_encoder_lora)
print_trainable_parameters(unet_lora)


class ImageDataset(Dataset):
    def __init__(self, root_dir, df, tokenizer, size = 224, center_crop = True):
        self.root_dir = root_dir
        self.files = df['File_name'].tolist()
        self.findings = df['text'].tolist()
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
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
        example["instance_prompt_ids"] = self.tokenizer(
            self.findings[idx],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        return example



# reports = reports[['File_name', 'text']]
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

valid_ds = ImageDataset(
    root_dir=Data_storage,
    df=valid_df,
    tokenizer=tokenizer,
)

test_ds = ImageDataset(
    root_dir=Data_storage,
    df=test_df,
    tokenizer=tokenizer,
)
print(f'n of working cpus: {os.cpu_count()}')
num_workers = 2
# DataLoader initialization
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers = num_workers)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers = num_workers)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE // 2, shuffle=False, num_workers = num_workers)

optimizer = torch.optim.Adam(list(unet_lora.parameters()) + list(text_encoder_lora.parameters()), lr=LR)

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
            
            # clear_cache()
            torch.cuda.empty_cache()


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

                self.total_step += 1

                #######################################################################
                # check number of expandable LoRA
                #######################################################################
                #### Commenting the expandable lora's 
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


                if self.total_step % self.per_iter_valid == 0:

                    valid_rmse = self.valid()

                    if valid_rmse <= min([x for x in trainer.result_df['Valid Loss'] if x != ' --- '] + [1.0]):

                        self.best_text_encoder = copy.deepcopy(self.text_encoder).cpu()
                        self.best_unet = copy.deepcopy(self.unet).cpu()

                        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model')
                        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
                        check_and_make_folder(f'{save_result_path}/{folder_name}/trained_model/final_Text')

                        self.unet.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Unet')
                        self.text_encoder.save_pretrained(f'{save_result_path}/{folder_name}/trained_model/final_Text')


                    self.result_df.loc[len(self.result_df)] = [epoch, self.total_step, np.round(np.mean(recorded_loss), 4), np.round(valid_rmse, 4), self.added_rank,  self.trainable_percentage(self.unet), self.trainable_percentage(self.text_encoder)]

                    # self._display_id.update(self.result_df)

                    print(self.result_df)
                    recorded_loss = []

                if self.total_step % self.log_period == 0:

                    self.result_df.loc[len(self.result_df)] = [epoch, self.total_step, np.round(np.mean(recorded_loss), 4), ' --- ', self.added_rank,  self.trainable_percentage(self.unet), self.trainable_percentage(self.text_encoder)]

                    print(self.result_df)  #
                    self.result_df.to_csv(f'{save_result_path}/{folder_name}/results.csv')




trainer = Trainer(
    vae = vae,
    unet = unet_lora,
    text_encoder = text_encoder_lora,
    noise_scheduler = noise_scheduler,
    optimizer = optimizer,
    train_dl = train_loader,
    test_dl = valid_loader,
    total_epoch = 10,
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

for place in range(len(test_df)):
    temp_prompts = test_df.text.iloc[place]

    temp = new_pipe(temp_prompts, height = 224, width = 224).images[0]

    temp.save(f'{save_result_path}/{folder_name}/test_images/{place}.png')

    display(temp)