# use idel gpu
# it's better to use enviroment variable
# if you want to use multiple gpus, please
# modify hyperparameters at the same time
# And Make Sure Your Pytorch Version >= 1.0.1
import os
import sys
from models import *

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
from metrics import DomainFD
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
run_name = "SR GAN"
if n_gpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if n_gpu == 2:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
if n_gpu == 4:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

device = torch.device('cuda:0')

base_lr = 0.001
# Original Learning Rate
learning_rate = {(25, 8): base_lr, (50, 16): base_lr, (100,32): base_lr, (200, 64): base_lr, (400, 128): base_lr, (800, 256): base_lr}
if n_gpu == 1:
    batch_size = {(25, 8): 64, (50, 16): 32, (100, 32): 2, (200, 64): 2, (400, 128): 2, (800, 256): 2}
if n_gpu == 2:
    batch_size = {(25, 8): 512, (50, 16): 512, (100, 32): 24, (200, 64): 16, (400, 128): 4, (800, 256): 4}
elif n_gpu == 4:
    batch_size = {(25, 8): 512, (50, 16): 360, (100, 32): 360, (200, 64): 72, (400, 128): 32, (800, 256): 6}
mini_batch_size = 8

num_workers = {(200, 64): 8, (400, 128): 4, (800, 256): 2}
max_workers = 16


lambda1 = 0.01
lambda2 = 0.01

n_fc = 8
dim_latent = 512
dim_input = (25, 8)
# number of samples to show before doubling resolution
# n_sample = 600_000
n_sample = 400_000
# number of samples train model in total
n_sample_total = 10_000_000
DGR = 1
n_show_loss = 1
n_save_im = 10
n_checkpoint = 100
step = 0  # Train from (8 * 8)
max_step = 5
style_mixing = []  # Waiting to implement

if n_gpu == 1 or n_gpu == 2:
    data_path = "./data"
elif n_gpu == 4:
    data_path = "./data"

save_im_path = "./g_z/" + run_name
if n_gpu == 1 or n_gpu == 2:
    save_checkpoints_path = "./checkpoints/" + run_name
elif n_gpu == 4:
    save_checkpoints_path = "/hpf/largeprojects/agoldenb/lechang/" + run_name

if n_gpu == 1 or n_gpu == 2:
    ae_dir = "./ae_checkpoints/ae-9600.pth"
else:
    ae_dir = "./ae-9600.pth"

# load_checkpoint = "/hpf/largeprojects/agoldenb/lechang/trained-1600.pth"
load_checkpoint = "no" # restart

if n_gpu == 1 or n_gpu == 2:
    wandb.init(project="mri_gan_cancer", name=run_name)
elif n_gpu == 4:
    wandb.init(project="mri_gan_cancer", name=run_name, dir="/hpf/largeprojects/agoldenb/lechang/")


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=str, default=str(batch_size), metavar='N',
                     help='')
parser.add_argument('--lr', type=str, default=str(learning_rate), metavar='N',
                     help='')
parser.add_argument('--data_path', type=str, default=data_path, metavar='N',
                     help='')
parser.add_argument('--g_z_path', type=str, default=save_im_path, metavar='N',
                     help='')
parser.add_argument('--checkpoints_path', type=str, default=save_checkpoints_path, metavar='N',
                     help='')
parser.add_argument('--load_checkpoint', type=str, default=load_checkpoint, metavar='N',
                     help='')
parser.add_argument('--lambda1', type=str, default=lambda1, metavar='N',
                     help='')
parser.add_argument('--lambda2', type=str, default=lambda2, metavar='N',
                     help='')

args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables


os.makedirs(args.g_z_path, exist_ok=True)
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
d_losses = [float('inf')]
g_losses = [float('inf')]
fd = [float('inf')]


def random_mix_steps():
    return list(range(random.randint(0, 10)))

def set_grad_flag(module, flag):
    for p in module.parameters():
        p.requires_grad = flag


def reset_LR(optimizer, lr):
    for pam_group in optimizer.param_groups:
        mul = pam_group.get('mul', 1)
        pam_group['lr'] = lr * mul


# Gain sample
def gain_sample(batch_size, image_size=(25,8)):
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


def sample_data(data_loader, origin_loader):
    try:
        # Try to read next image
        real_image = next(data_loader)

    except (OSError, StopIteration):
        # Dataset exhausted, train from the first image
        data_loader = iter(origin_loader)
        real_image = next(data_loader)
    return real_image


def discriminate(discriminator, real_image, step, alpha, n_gpu):
    if n_gpu > 1:
        predict = nn.parallel.data_parallel(discriminator, (real_image, step, alpha), range(n_gpu))
    else:
        predict = discriminator(real_image, step, alpha)
    return predict


def generate(generator, step, alpha, mix_steps, resolution, n_gpu, latent_w=None):
    if latent_w is None:
        latent_w = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device),
                    torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]
        bs = batch_size.get(resolution, mini_batch_size)
    else:
        bs = latent_w[0].shape[0]

    noise = []
    for m in range(step + 1):
        size_x = 25 * 2 ** m  # Due to the upsampling, size of noise will grow
        size_y = 8 * 2 ** m
        noise.append(torch.randn((bs, 1, size_x, size_y), device=device))

    if n_gpu > 1:
        fake_image = nn.parallel.data_parallel(generator, (latent_w, step, alpha, noise, mix_steps),
                                               range(n_gpu))
    else:
        fake_image = generator(latent_w, step, alpha, noise, mix_steps)

    return fake_image


def encode(encoder, image, step, alpha, n_gpu):
    if n_gpu > 1:
        encoding = nn.parallel.data_parallel(encoder, (image, step, alpha), range(n_gpu))
    else:
        encoding = encoder(image, step, alpha)
    return encoding


# Train function
def train(generator, discriminator, autoencoder, g_optim, d_optim, step, iteration=0, startpoint=0, used_sample=0,
          d_losses=[], g_losses=[],alpha=0):



    resolution = (25 * 2 ** step, 8 * 2 ** step)

    origin_loader = gain_sample(batch_size.get(resolution, mini_batch_size), resolution)
    data_loader = iter(origin_loader)

    ae_step = 4
    ae_resolution = (25 * 2 ** ae_step, 8 * 2 ** ae_step)
    fd_calculator = DomainFD(autoencoder, ae_resolution, device=device)

    ae_data_loader = iter(gain_sample(359, ae_resolution))
    fd_calculator.fit_real_data(ae_data_loader)

    reset_LR(g_optim, learning_rate.get(resolution, base_lr))
    reset_LR(d_optim, learning_rate.get(resolution, base_lr))


    progress_bar = tqdm(total=n_sample_total, initial=used_sample)
    # Train
    while used_sample < n_sample_total:

        iteration += 1
        alpha = min(1, alpha + batch_size.get(resolution, mini_batch_size) / (n_sample))

        if (used_sample - startpoint) > n_sample and step < max_step:
            step += 1
            print("Now on step", step)
            alpha = 0
            startpoint = used_sample

            resolution = (25 * 2 ** step, 8 * 2 ** step)

            # Avoid possible memory leak
            del origin_loader

            # Change batch size
            # apply resizing
            origin_loader = gain_sample(batch_size.get(resolution, mini_batch_size), resolution)

            data_loader = iter(origin_loader)

            reset_LR(g_optim, learning_rate.get(resolution, base_lr))
            reset_LR(d_optim, learning_rate.get(resolution, base_lr))



        # D Update
        real_image = sample_data(data_loader, origin_loader)

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)

        real_predict = discriminate(discriminator, real_image, step, alpha, n_gpu)
        real_loss = nn.functional.softplus(-real_predict).mean()


        fake_image = generate(generator, step, alpha, random_mix_steps(), resolution, n_gpu)
        fake_predict = discriminate(discriminator, fake_image, step, alpha, n_gpu)

        fake_loss = nn.functional.softplus(fake_predict).mean()

        d_loss = real_loss + fake_loss

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        del real_image, real_predict, real_loss, fake_image, fake_predict, fake_loss

        # G Update
        fake_image = generate(generator, step, alpha, random_mix_steps(), resolution, n_gpu)
        fake_predict = discriminate(discriminator, fake_image, step, alpha, n_gpu)
        fake_loss = nn.functional.softplus(-fake_predict).mean()

        g_optim.zero_grad()
        fake_loss.backward()
        g_optim.step()


        if iteration % n_show_loss == 0:
            g_losses.append(fake_loss.item())
            d_losses.append(d_loss.item())
            # print(fd_calculator.calculate_fd(fake_image))
            fd.append(fd_calculator.calculate_fd(fake_image))

            wandb.log({"G Loss": g_losses[-1],
                       "D Loss": d_losses[-1],
                       "Domain FD": fd[-1],
                       "Images Shown": n_sample
                       },
                      step=iteration)

        if iteration % n_save_im == 0:
            imsave(fake_image.data.cpu(), iteration)

        del fake_image, fake_loss




        if iteration % n_checkpoint == 0:
            os.makedirs(save_checkpoints_path, exist_ok=True)
            # Save the model every 50 iterations
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample, alpha),
                'd_losses': d_losses,
                'g_losses': g_losses,

            }, f'{save_checkpoints_path}/trained-{iteration}.pth')
            wandb.save(f'{save_checkpoints_path}/trained-{iteration}.pth')
            print(f' Model successfully saved.')


        progress_bar.set_description(
            (f'Resolution: {resolution[0]}*{resolution[1]}  D_Loss: {d_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}')
        )


# Create models
generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
discriminator = Discriminator(sr=True).to(device)

# resize images to fit this size
autoencoder = AutoEncoder(step=4).to(device)
ae_checkpoint = torch.load(ae_dir)
autoencoder.load_state_dict(ae_checkpoint['ae'])
autoencoder.eval()

wandb.watch((generator, discriminator))

# Optimizers
g_optim = optim.Adam([{
    'params': generator.convs.parameters(),
    'lr': base_lr
}, {
    'params': generator.to_rgbs.parameters(),
    'lr': base_lr
}, {
    'params': generator.fcs.parameters(),
    'lr': base_lr,
    'mul': 0.01
}], lr=base_lr, betas=(0.0, 0.99))
d_optim = optim.Adam(discriminator.parameters(), lr=base_lr, betas=(0.0, 0.99))

if is_continue:
    if os.path.exists(load_checkpoint):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(load_checkpoint)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator1'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
        g_losses = checkpoint.get('g_losses', [float('inf')])
        d_losses = checkpoint.get('d_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

generator.train()
discriminator.train()



train(generator, discriminator, autoencoder, g_optim, d_optim, step, iteration, startpoint,
                           used_sample, d_losses, g_losses, alpha)#, method="MDGAN")