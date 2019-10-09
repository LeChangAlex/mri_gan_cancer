# use idel gpu
# it's better to use enviroment variable
# if you want to use multiple gpus, please
# modify hyperparameters at the same time
# And Make Sure Your Pytorch Version >= 1.0.1
import os
import sys
from models import StyleBased_Generator
from models import Discriminator
from models import Encoder

from MRIDataset import MRIDataset
import time

# Import necessary modules
import numpy as np
from tqdm import tqdm
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch

#---------------------------------------------------
# Copied module to solve shared memory conflict trouble
# 5/15: No using shared memory
# 5/23: Seems it does not work :(
import sys
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

# Weights and biases logging
import wandb
import argparse
import json

n_gpu = 1
run_name = "AE"
num_workers = {(200, 64): 8, (400, 128): 4, (800, 256): 2}
max_workers = 16

n_sample_total = 1_000_000
step = 4
batch_size = 256

if n_gpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if n_gpu == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

device = torch.device('cuda:0')

learning_rate = 0.001

if n_gpu == 1:
    data_path = "./data"
elif n_gpu == 4:
    data_path = "./data"

save_im_path = "./ae_reconstruct/" + run_name
if n_gpu == 1:
    save_checkpoints_path = "./ae_checkpoints/" + run_name
elif n_gpu == 4:
    save_checkpoints_path = "/hpf/largeprojects/agoldenb/lechang/ae/" + run_name

# load_checkpoint = "/hpf/largeprojects/agoldenb/lechang/trained-1600.pth"
load_checkpoint = "no" # restart

DGR = 1
n_show_loss = 1
n_save_im = 10
n_checkpoint = 1600

wandb.init(project="mri_gan_cancer", name=run_name)

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=str, default=str(batch_size), metavar='N',
                     help='')
parser.add_argument('--lr', type=str, default=str(learning_rate), metavar='N',
                     help='')
parser.add_argument('--data_path', type=str, default=data_path, metavar='N',
                     help='')
parser.add_argument('--rec_im_path', type=str, default=save_im_path, metavar='N',
                     help='')
parser.add_argument('--checkpoints_path', type=str, default=save_checkpoints_path, metavar='N',
                     help='')
parser.add_argument('--load_checkpoint', type=str, default=load_checkpoint, metavar='N',
                     help='')

args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables


os.makedirs(args.rec_im_path, exist_ok=True)
os.makedirs(args.checkpoints_path, exist_ok=True)



# Used to continue training from last checkpoint
iteration = 0
startpoint = 0
used_sample = 0
alpha = 0

# How to start training?
# True for start from saved model
# False for retrain from the very beginning
is_continue = True
ae_losses = []
#

class SConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.conv = nn.Conv2d(*args, **kwargs)
        self.conv.weight.data.normal_()
        self.conv.bias.data.zero_()



    def forward(self, x):
        return self.conv(x)

class SDeconv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.deconv = nn.ConvTranspose2d(*args, **kwargs)
        self.deconv.weight.data.normal_()
        self.deconv.bias.data.zero_()


    def forward(self, x):
        return self.deconv(x)


class AutoEncoder(nn.Module):
    def __init__(self, step):
        super().__init__()

        self.relu = nn.ReLU()

        self.conv_layers = nn.ModuleList([])

        # Encoder
        # (800, 256) -> (400, 128) -> (200, 64) -> (100, 32) -> (50, 16) -> (25, 8) ->
        self.conv_layers.append(SConv2d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=3,
                                      padding=1,
                                      stride=(2, 2)))
        for i in range(step):
            self.conv_layers.append(nn.Sequential(
                SConv2d(in_channels=16*(2**i),
                          out_channels=16*(2**(i+1)),
                          kernel_size=3,
                          padding=1,
                          stride=(2, 2))
            ))
        # Decoder
        for i in range(step):
            self.conv_layers.append(nn.Sequential(
                SDeconv2d(in_channels=16 * (2 ** (step - i)),
                                   out_channels=16 * (2 ** (step - i - 1)),
                                   kernel_size=4,
                                   padding=1,
                                   stride=(2, 2))
            ))
        self.conv_layers.append(SDeconv2d(in_channels=16,
                                              out_channels=1,
                                              kernel_size=4,
                                              padding=1,
                                              stride=(2, 2)))

    def forward(self, *input):

        result = input[0]
        for i in range(len(self.conv_layers) - 1):
            result = self.conv_layers[i](result)
            result = self.relu(result)

        result = self.conv_layers[-1](result)
        # result = torch.sigmoid(result)

        return result


def reset_LR(optimizer, lr):
    for pam_group in optimizer.param_groups:
        mul = pam_group.get('mul', 1)
        pam_group['lr'] = lr * mul


# Gain sample
def gain_sample(batch_size, image_size=(25, 8)):
    loader = DataLoader(MRIDataset(
        csv_file="./annotations_slices_medium.csv",
        root_dir=args.data_path,
        transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    ), shuffle=True, batch_size=batch_size,
                        num_workers=num_workers.get(image_size, max_workers))

    return loader


f = plt.figure()
def imsave(tensor, i):
    wandb.log({"G(z)":[wandb.Image(tensor[i][0], mode="F") for i in range(min(tensor.shape[0], 10))]}, step=i)


# Train function
def train(ae, ae_optim, step, iteration=0, startpoint=0, used_sample=0, ae_losses=[]):

    criterion = nn.BCEWithLogitsLoss()
    origin_loader = gain_sample(batch_size=batch_size, image_size=(800, 256))
    data_loader = iter(origin_loader)

    progress_bar = tqdm(total=n_sample_total, initial=used_sample)
    # Train
    while used_sample < n_sample_total:
        iteration += 1

        try:
            # Try to read next image
            real_image = next(data_loader)

        except (OSError, StopIteration):
            # Dataset exhausted, train from the first image
            data_loader = iter(origin_loader)
            real_image = next(data_loader)

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)



        # real_image.requires_grad = False
        if n_gpu > 1:
            reconstruction = nn.parallel.data_parallel(ae, (real_image), range(n_gpu))
        else:
            reconstruction = ae(real_image)

        # loss = reconstruction.sum()
        # loss = torch.nn.functional.binary_cross_entropy(reconstruction, real_image).mean()
        loss = torch.nn.functional.mse_loss(reconstruction, real_image)
        # loss = ((reconstruction - real_image)**2).mean()
        # loss = criterion(reconstruction, real_image)
        ae_optim.zero_grad()
        loss.backward()
        # print(list(ae.parameters())[0].grad)

        ae_optim.step()

        if iteration % n_show_loss == 0:
            ae_losses.append(loss.item())
            wandb.log({"AE Loss": ae_losses[-1],
                       },
                      step=iteration)
            # TODO: add other metrics to log (FID, ...)

        if iteration % n_save_im == 0:
            imsave(reconstruction.data.cpu(), iteration)



        if iteration % n_checkpoint == 0:
            os.makedirs(save_checkpoints_path, exist_ok=True)
            # Save the model every 50 iterations
            torch.save({
                'ae': ae.state_dict(),
                'ae_optim': ae_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample),
                'ae_losses': ae_losses
            }, f'{save_checkpoints_path}/trained-{iteration}.pth')
            wandb.save(f'{save_checkpoints_path}/trained-{iteration}.pth')
            print(f' Model successfully saved.')


        progress_bar.set_description(
            (f'Resolution: AE_Loss: {ae_losses[-1]:.4f}')
        )


# Create models
ae = AutoEncoder(step).to(device)

wandb.watch(ae)

# Optimizers
ae_optim = optim.Adam(ae.parameters(), lr=learning_rate, betas=(0.0, 0.99))

if is_continue:
    if os.path.exists(load_checkpoint):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(load_checkpoint)
        ae.load_state_dict(checkpoint['ae'])
        ae_optim.load_state_dict(checkpoint['ae_optim'])
        step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
        d_losses = checkpoint.get('d_losses', [float('inf')])
        g_losses = checkpoint.get('g_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

ae.train()

train(ae, ae_optim, step, iteration, startpoint,
                           used_sample, ae_losses)#, method="MDGAN")
