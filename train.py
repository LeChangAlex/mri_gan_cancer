# use idel gpu
# it's better to use enviroment variable
# if you want to use multiple gpus, please
# modify hyperparameters at the same time
# And Make Sure Your Pytorch Version >= 1.0.1
import os
import sys
from models import StyleBased_Generator
from models import Discriminator
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
n_gpu = 2
device = torch.device('cuda:0')

# Original Learning Rate
learning_rate = {(100,32): 0.001, (200, 64): 0.001, (400, 128): 0.001, (800, 256): 0.001}
# For anime only
# learning_rate     = {512: 0.0015, 1024: 0.002}
batch_size = {(25, 8): 128, (50, 16): 128, (100, 32): 64, (200, 64): 10, (400, 128): 4, (800, 256): 4}
mini_batch_size = 8
batch_size_1gpu = {4: 256, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
mini_batch_size = 8
batch_size_4gpus = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
mini_batch_size_4 = 16
batch_size_8gpus = {4: 512, 8: 256, 16: 128, 32: 64}
mini_batch_size_8 = 32
# Comment line below if you don't meet the problem of 'shared memory conflict'
num_workers = {(200, 64): 8, (400, 128): 4, (800, 256): 2}
max_workers = 16
n_fc = 8
dim_latent = 512
dim_input = (25, 8)
# number of samples to show before dowbling resolution
n_sample = 600_000
# number of samples train model in total
n_sample_total = 10_000_000
DGR = 1
n_show_loss = 10
step = 0  # Train from (8 * 8)
max_step = 6
style_mixing = []  # Waiting to implement
image_folder_path = './wbmri_slices_medium'
save_folder_path = './g_z/08_22_2019/'

low_steps = [0, 1, 2]
# style_mixing    += low_steps
mid_steps = [3, 4, 5]
# style_mixing    += mid_steps
hig_steps = [6, 7, 8]
# style_mixing    += hig_steps

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
        csv_file="../annotations_slices_medium.csv",
        root_dir="../wbmri_slices_medium",
        transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    ), shuffle=True, batch_size=batch_size,
                        num_workers=num_workers.get(image_size, max_workers))

    return loader

f = plt.figure()
def imsave(tensor, i):
    try:
        f.add_subplot(1, 6, 1)
        plt.imshow(tensor[0][0])
        f.add_subplot(1, 6, 2)
        plt.imshow(tensor[1][0])
        f.add_subplot(1, 6, 3)
        plt.imshow(tensor[2][0])
        f.add_subplot(1, 6, 4)
        plt.imshow(tensor[3][0])
        f.add_subplot(1, 6, 5)
        plt.imshow(tensor[4][0])
        f.add_subplot(1, 6, 6)
        plt.imshow(tensor[5][0])
        grid = tensor[0][0]
        plt.show(block=False)
        # plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
        # plt.imshow()
        plt.pause(0.001)
        plt.clf()
        grid.clamp_(-1, 1).add_(1).div_(2)
        # Add 0.5 after normalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(f'{save_folder_path}sample-iter{i}.png')
    except:
        print("Less than 6 images in the batch to plot.")
        grid = tensor[0][0]
        grid.clamp_(-1, 1).add_(1).div_(2)
        # Add 0.5 after normalizing to [0, 255] to round to nearest integer
        ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(f'{save_folder_path}sample-iter{i}.png')
        pass

# Train function
def train(generator, discriminator, g_optim, d_optim, step, iteration=0, startpoint=0, used_sample=0,
          d_losses=[], g_losses=[], alpha=0):
    resolution = (25 * 2 ** step, 8 * 2 ** step)

    origin_loader = gain_sample(batch_size.get(resolution, mini_batch_size), resolution)
    data_loader = iter(origin_loader)

    reset_LR(g_optim, learning_rate.get(resolution, 0.001))
    reset_LR(d_optim, learning_rate.get(resolution, 0.001))
    progress_bar = tqdm(total=n_sample_total, initial=used_sample)
    # Train
    while used_sample < n_sample_total:
        iteration += 1
        alpha = min(1, alpha + batch_size.get(resolution, mini_batch_size) / (n_sample))

        if (used_sample - startpoint) > n_sample and step < max_step:
            step += 1
            alpha = 0
            startpoint = used_sample

            resolution = (25 * 2 ** step, 8 * 2 ** step)

            # Avoid possible memory leak
            del origin_loader
            del data_loader

            # Change batch size
            origin_loader = gain_sample(batch_size.get(resolution, mini_batch_size), resolution)
            data_loader = iter(origin_loader)

            reset_LR(g_optim, learning_rate.get(resolution, 0.001))
            reset_LR(d_optim, learning_rate.get(resolution, 0.001))

        try:
            # Try to read next image
            real_image= next(data_loader)

        except (OSError, StopIteration):
            # Dataset exhausted, train from the first image
            data_loader = iter(origin_loader)
            real_image = next(data_loader)

        # Count used sample
        used_sample += real_image.shape[0]
        progress_bar.update(real_image.shape[0])

        # Send image to GPU
        real_image = real_image.to(device)

        # D Module ---
        # Train discriminator first
        discriminator.zero_grad()
        set_grad_flag(discriminator, True)
        set_grad_flag(generator, False)

        # Real image predict & backward
        # We only implement non-saturating loss with R1 regularization loss
        real_image.requires_grad = True
        if n_gpu > 1:
            real_predict = nn.parallel.data_parallel(discriminator, (real_image, step, alpha), range(n_gpu))
        else:
            real_predict = discriminator(real_image, step, alpha)
        real_predict = nn.functional.softplus(-real_predict).mean()
        real_predict.backward(retain_graph=True)

        grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real
        grad_penalty_real.backward()

        # Generate latent code
        latent_w1 = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device),
                     torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]
        latent_w2 = [torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device),
                     torch.randn((batch_size.get(resolution, mini_batch_size), dim_latent), device=device)]

        noise_1 = []
        noise_2 = []
        for m in range(step + 1):
            size_x = 25 * 2 ** m  # Due to the upsampling, size of noise will grow
            size_y = 8 * 2 ** m
            noise_1.append(torch.randn((batch_size.get(resolution, mini_batch_size), 1, size_x, size_y), device=device))
            noise_2.append(torch.randn((batch_size.get(resolution, mini_batch_size), 1, size_x, size_y), device=device))

        # Generate fake image & backward
        if n_gpu > 1:
            fake_image = nn.parallel.data_parallel(generator, (latent_w1, step, alpha, noise_1, random_mix_steps()), range(n_gpu))
            fake_predict = nn.parallel.data_parallel(discriminator, (fake_image, step, alpha), range(n_gpu))
        else:
            fake_image = generator(latent_w1, step, alpha, noise_1, random_mix_steps())
            fake_predict = discriminator(fake_image, step, alpha)

        fake_predict = nn.functional.softplus(fake_predict).mean()
        fake_predict.backward()

        if iteration % n_show_loss == 0:
            d_losses.append((real_predict + fake_predict).item())

        # D optimizer step
        d_optim.step()

        # Avoid possible memory leak
        del grad_penalty_real, grad_real, fake_predict, real_predict, fake_image, real_image, latent_w1

        # G module ---
        if iteration % DGR != 0: continue
        # Due to DGR, train generator
        generator.zero_grad()
        set_grad_flag(discriminator, False)
        set_grad_flag(generator, True)

        if n_gpu > 1:
            fake_image = nn.parallel.data_parallel(generator, (latent_w2, step, alpha, noise_2, random_mix_steps()), range(n_gpu))
            fake_predict = nn.parallel.data_parallel(discriminator, (fake_image, step, alpha), range(n_gpu))
        else:
            fake_image = generator(latent_w2, step, alpha, noise_2, random_mix_steps())
            fake_predict = discriminator(fake_image, step, alpha)
        fake_predict = nn.functional.softplus(-fake_predict).mean()
        fake_predict.backward()
        g_optim.step()

        if iteration % n_show_loss == 0:
            g_losses.append(fake_predict.item())
            imsave(fake_image.data.cpu(), iteration)

        # Avoid possible memory leak
        del fake_predict, fake_image, latent_w2

        if iteration % 250 == 0:
            # Save the model every 50 iterations
            torch.save({
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optim': g_optim.state_dict(),
                'd_optim': d_optim.state_dict(),
                'parameters': (step, iteration, startpoint, used_sample, alpha),
                'd_losses': d_losses,
                'g_losses': g_losses
            }, 'networks/trained_{}.pth'.format(iteration))
            print(f'Model successfully saved.')

        progress_bar.set_description((
                                         f'Resolution: {resolution[0]}*{resolution[1]}  D_Loss: {d_losses[-1]:.4f}  G_Loss: {g_losses[-1]:.4f}  Alpha: {alpha:.4f}'))
    torch.save({
        'generator': generator.state_dict(),
        'discriminator': discriminator.state_dict(),
        'g_optim': g_optim.state_dict(),
        'd_optim': d_optim.state_dict(),
        'parameters': (step, iteration, startpoint, used_sample, alpha),
        'd_losses': d_losses,
        'g_losses': g_losses
    }, 'networks/trained.pth')
    print(f'Final model successfully saved.')
    return d_losses, g_losses

# Create models
generator = StyleBased_Generator(n_fc, dim_latent, dim_input).to(device)
discriminator = Discriminator().to(device)

# Optimizers
g_optim = optim.Adam([{
    'params': generator.convs.parameters(),
    'lr': 0.001
}, {
    'params': generator.to_rgbs.parameters(),
    'lr': 0.001
}], lr=0.001, betas=(0.0, 0.99))
g_optim.add_param_group({
    'params': generator.fcs.parameters(),
    'lr': 0.001 * 0.01,
    'mul': 0.01
})
d_optim = optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.0, 0.99))

if is_continue:
    if os.path.exists('networks/trained_279750.pth'):
        # Load data from last checkpoint
        print('Loading pre-trained model...')
        checkpoint = torch.load('networks/trained_279750.pth')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optim.load_state_dict(checkpoint['g_optim'])
        d_optim.load_state_dict(checkpoint['d_optim'])
        step, iteration, startpoint, used_sample, alpha = checkpoint['parameters']
        d_losses = checkpoint.get('d_losses', [float('inf')])
        g_losses = checkpoint.get('g_losses', [float('inf')])
        print('Start training from loaded model...')
    else:
        print('No pre-trained model detected, restart training...')

generator.train()
discriminator.train()
d_losses, g_losses = train(generator, discriminator, g_optim, d_optim, step, iteration, startpoint,
                           used_sample, d_losses, g_losses, alpha)
