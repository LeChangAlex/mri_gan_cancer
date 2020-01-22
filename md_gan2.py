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
from arguments import get_args
from torchvision.utils import save_image


device = torch.device('cuda:0')



mini_batch_size = 8

num_workers = {(200, 64): 8, (400, 128): 4, (800, 256): 2}
max_workers = 16


n_fc = 8
dim_latent = 512
dim_input = (25, 8)
# number of samples to show before doubling resolution
# n_sample = 600_000
# number of samples train model in total
n_sample_total = 10_000_000
DGR = 1
n_show_loss = 1
n_save_im = 1
n_checkpoint = 1000
step = 0  # Train from (8 * 8)
max_step = 5
style_mixing = []  # Waiting to implement

learning_rate = 0.001


# /hpf/largeprojects/agoldenb/lechang/

args = get_args()
batch_size = [int(bs) for bs in args.batch_size.split(",")]

wandb.init(project="mri_gan_cancer", name=args.run_name, dir=args.wandb_dir)

wandb.config.update(args) # adds all of the arguments as config variables



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
    save_image(tensor, os.path.join(args.g_z_path, args.run_name, "step {}.png".format(i)), nrow=1, padding=0, normalize=True)


def sample_data(data_loader, origin_loader):
    try:
        # Try to read next image
        real_image = next(data_loader)

    except (OSError, StopIteration):
        # Dataset exhausted, train from the first image
        data_loader = iter(origin_loader)
        real_image = next(data_loader)
    return real_image


def discriminate(discriminator, real_image, step, alpha, std, n_gpu):
    if args.n_gpu > 1:
        predict = nn.parallel.data_parallel(discriminator, (real_image, step, alpha, std), range(n_gpu))
    else:
        predict = discriminator(real_image, step, alpha, std)

    return predict


def generate(generator, step, alpha, mix_steps, resolution, n_gpu, latent_w=None):
    if latent_w is None:
        latent_w = [torch.randn((batch_size[step], dim_latent), device=device),
                    torch.randn((batch_size[step], dim_latent), device=device)]
        bs = batch_size[step]
    else:
        bs = latent_w[0].shape[0]

    noise = []
    for m in range(step + 1):
        size_x = 25 * 2 ** m  # Due to the upsampling, size of noise will grow
        size_y = 8 * 2 ** m
        noise.append(torch.randn((bs, 1, size_x, size_y), device=device))

    if args.n_gpu > 1:
        fake_image = nn.parallel.data_parallel(generator, (latent_w, step, alpha, noise, mix_steps),
                                               range(n_gpu))
    else:
        fake_image = generator(latent_w, step, alpha, noise, mix_steps)

    return fake_image


def encode(encoder, image, step, alpha, n_gpu):
    if args.n_gpu > 1:
        encoding = nn.parallel.data_parallel(encoder, (image, step, alpha), range(args.n_gpu))
    else:
        encoding = encoder(image, step, alpha)
    return encoding


# Train function
def train(generator, discriminator1, discriminator2, encoder, autoencoder, g_optim, d1_optim, d2_optim, e_optim, step, iteration=0, startpoint=0, used_sample=0,
          d1_losses=[], d2_losses=[], e_losses=[], g_losses=[],alpha=0):

    std = 0.2

    resolution = (25 * 2 ** step, 8 * 2 ** step)

    origin_loader = gain_sample(batch_size[step], resolution)
    data_loader1 = iter(origin_loader)
    data_loader2 = iter(origin_loader)


    reset_LR(g_optim, args.lr)
    reset_LR(d1_optim, args.lr)
    reset_LR(d2_optim, args.lr)
    reset_LR(e_optim, args.lr)


    progress_bar = tqdm(total=n_sample_total, initial=used_sample)
    # Train
    while used_sample < n_sample_total:

        # done 800 x 256 step
        if used_sample > 3_600_000:
            std = args.instance_noise - (used_sample - 3_600_000) * args.instance_noise / 600_000
            std = max(std, 0)


        iteration += 1
        alpha = min(1, alpha + batch_size[step] / (args.n_sample))

        if (used_sample - startpoint) > args.n_sample and step < max_step:
            step += 1
            print("Now on step", step)
            alpha = 0
            startpoint = used_sample

            resolution = (25 * 2 ** step, 8 * 2 ** step)

            # Avoid possible memory leak
            del origin_loader

            # Change batch size
            # apply resizing
            origin_loader = gain_sample(batch_size[step], resolution)

            data_loader1 = iter(origin_loader)
            data_loader2 = iter(origin_loader)


        real_image = sample_data(data_loader1, origin_loader)

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)

        # Manifold step--------------------------------------------------
        # D Module ---
        # Train discriminator first
        discriminator1.zero_grad()

        d_real_image = discriminate(discriminator1, real_image, step, alpha, std, args.n_gpu)
        encoding = encode(encoder, real_image, step, alpha, args.n_gpu)
        ge_real_image = generate(generator, step, alpha, [], resolution, args.n_gpu, latent_w=[encoding])
        dge_real_image = discriminate(discriminator1, ge_real_image, step, alpha, std, args.n_gpu)

        d1_loss = nn.functional.softplus(-d_real_image).mean() \
                  + nn.functional.softplus(dge_real_image).mean()

        d1_optim.zero_grad()
        d1_loss.backward(retain_graph=True)
        d1_optim.step()

        # Euclidean distance
        l2_loss = torch.sum((real_image - ge_real_image) ** 2, 1)
        l2_loss = torch.sum(l2_loss, 1).mean()

        e_optim.zero_grad()
        l2_loss.backward(retain_graph=True)
        e_optim.step()

        generator.zero_grad()
        g_loss = args.lambda1 * nn.functional.softplus(-dge_real_image).mean() + l2_loss

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if iteration % n_show_loss == 0:
            d1_losses.append(d1_loss.item())
            e_losses.append(l2_loss.item())

        del real_image, d_real_image, encoding, ge_real_image, dge_real_image, d1_loss, l2_loss, g_loss
        # grad_real = torch.autograd.grad(outputs=real_loss.sum(), inputs=real_image, create_graph=True)[0]
        # grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        # grad_penalty_real = 10 / 2 * grad_penalty_real
        # grad_penalty_real.backward()

        real_image = sample_data(data_loader2, origin_loader)

        # Send image to GPU
        real_image = real_image.to(device)

        encoding = encode(encoder, real_image, step, alpha, args.n_gpu)
        ge_real_image = generate(generator, step, alpha, [], resolution, args.n_gpu, latent_w=[encoding])
        dge_real_image = discriminate(discriminator2, ge_real_image, step, alpha, std, args.n_gpu)

        fake_image = generate(generator, step, alpha, random_mix_steps(), resolution, args.n_gpu)
        d_fake_image = discriminate(discriminator2, fake_image, step, alpha, std, args.n_gpu)

        d2_loss = nn.functional.softplus(-dge_real_image).mean() + nn.functional.softplus(d_fake_image).mean()

        d2_optim.zero_grad()
        d2_loss.backward(retain_graph=True)
        d2_optim.step()

        g_loss = nn.functional.softplus(-d_fake_image).mean()

        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if iteration % n_show_loss == 0:
            g_losses.append(g_loss.item())
            d2_losses.append(d2_loss.item())
            # print(fd_calculator.calculate_fd(fake_image))

            wandb.log({"G Loss": g_losses[-1],
                       "D1 Loss": d1_losses[-1],
                       "D2 Loss": d2_losses[-1],
                       "E Loss": e_losses[-1],
                       # "Domain FD": fd[-1],
                       "Images Shown": used_sample
                       },
                      step=iteration)

        if iteration % n_save_im == 0:
            imsave(fake_image.data.cpu(), iteration)

        del real_image, encoding, ge_real_image, dge_real_image, fake_image, d_fake_image, d2_loss


        if iteration % n_checkpoint == 0:
            # Save the model every 50 iterations
            torch.save({
                'generator': generator.state_dict(),
                'discriminator1': discriminator1.state_dict(),
                'discriminator2': discriminator2.state_dict(),
                'encoder': encoder.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd1_optim': d1_optim.state_dict(),
                'd2_optim': d2_optim.state_dict(),
                'e_optim': e_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample, alpha),
                'd1_losses': d1_losses,
                'd2_losses': d2_losses,
                'g_losses': g_losses,
                'e_losses': e_losses

            }, f'{args.save_checkpoints_path}/trained-{iteration}.pth')
            wandb.save(f'{args.save_checkpoints_path}/trained-{iteration}.pth')
            print(f' Model successfully saved.')



        progress_bar.set_description(
            (f'Resolution: {resolution[0]}*{resolution[1]}  D1_Loss: {d1_losses[-1]:.4f}  D2_Loss: {d2_losses[-1]:.4f}  E_Loss: {e_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}')
        )


# Create models
generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
discriminator1 = DiscriminatorLite().to(device)
discriminator2 = DiscriminatorLite().to(device)
encoder = EncoderLite().to(device)


autoencoder = None

wandb.watch((generator, discriminator1, discriminator2, encoder))

# Optimizers
g_optim = optim.Adam([{
    'params': generator.convs.parameters(),
    'lr': args.lr
}, {
    'params': generator.to_rgbs.parameters(),
    'lr': args.lr
}, {
    'params': generator.fcs.parameters(),
    'lr': args.lr,
    'mul': 0.01
}], lr=args.lr, betas=(0.0, 0.99))
d1_optim = optim.Adam(discriminator1.parameters(), lr=args.lr, betas=(0.0, 0.99))
d2_optim = optim.Adam(discriminator2.parameters(), lr=args.lr, betas=(0.0, 0.99))
e_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0.0, 0.99))

if is_continue:
    if os.path.exists(args.load_checkpoint):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load(args.load_checkpoint)
        generator.load_state_dict(checkpoint['generator'])
        discriminator1.load_state_dict(checkpoint['discriminator1'])
        discriminator2.load_state_dict(checkpoint['discriminator2'])
        encoder.load_state_dict(checkpoint['encoder'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d1_optim.load_state_dict(checkpoint['d1_optim'])
        d2_optim.load_state_dict(checkpoint['d2_optim'])
        e_optim.load_state_dict(checkpoint['e_optim'])
        step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
        d1_losses = checkpoint.get('d1_losses', [float('inf')])
        d2_losses = checkpoint.get('d2_losses', [float('inf')])
        g_losses = checkpoint.get('g_losses', [float('inf')])
        e_losses = checkpoint.get('e_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

generator.train()
discriminator1.train()
discriminator2.train()
encoder.train()

d1_losses = []
d2_losses = []
e_losses = []

train(generator, discriminator1, discriminator2, encoder, autoencoder, g_optim, d1_optim, d2_optim, e_optim, step, iteration, startpoint,
                           used_sample, d1_losses, d2_losses, e_losses, g_losses, alpha)#, method="MDGAN")
