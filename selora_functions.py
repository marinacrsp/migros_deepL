
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
    

def remove_param_from_optimizer(optim, param):
    for j in range(len(optim.param_groups)):
        optim_param_group_list = optim.param_groups[j]["params"]
        for i, optim_param in enumerate(optim_param_group_list):
            if param.shape == optim_param.shape and (param==optim_param).all():
                del optim.param_groups[j]["params"][i]




def set_Linear_SeLoRA(model, target_modules, rank):
    # works!
     # replace all linear layer (include q,k,v layer) into DyLoRA Layer.
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):

            LoRA_layer = Linear(
                  in_features = layer.in_features,
                  out_features = layer.out_features,
                  r = rank
            )

            LoRA_layer.weight = layer.weight
            LoRA_layer.weight.requires_grad = False
            LoRA_layer.bias = layer.bias
            if LoRA_layer.bias != None:
                LoRA_layer.bias.requires_grad = False

            pointing_layer = model
            #if len(target_modules) == 0:
            if False:
                if name.split('.')[-1] in target_modules:
                    for layer_name in name.split('.')[:-1]:
                        pointing_layer = getattr(pointing_layer, layer_name)
            else:
              if name.split('.')[-1] in target_modules:
                for layer_name in name.split('.')[:-1]:
                        pointing_layer = getattr(pointing_layer, layer_name)

                setattr(pointing_layer, name.split('.')[-1], LoRA_layer)
    return model



class ImageDataset(Dataset):
    def __init__(self, root_dir, df, tokenizer, size = 224, center_crop = True):
        self.root_dir = root_dir
        self.files = df['file_name'].tolist()
        self.findings = df['text'].tolist()
        self.tokenizer = tokenizer
        self.image_transforms = transforms.ToTensor()


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