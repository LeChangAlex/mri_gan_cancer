"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch

import numpy as np
from scipy import linalg
from scipy.misc import imread
from torch.nn.functional import adaptive_avg_pool2d

from train import *

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3
from model import Generator

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')


def get_activations(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(len(files))
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    # n_batches = len(files) // batch_size
    n_used_imgs = 356

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(0, n_used_imgs, args.batch_size):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)

        start = i
        end = min(356, start + batch_size)


        images = [np.load(str(f)).astype(np.float32)
                           for f in files[start:end]]


        # images = [imread(str(f)).astype(np.float32)
        #                    for f in files[start:end]]
        images_3ch = []
        for im in images:
            channels = np.stack((im,)*3, axis=-1).reshape(3,im.shape[1], im.shape[0])
            images_3ch.append(channels)
        images = np.array(images_3ch)
        images = images.transpose((0, 1, 3, 2))

        # # Reshape to (n_images, 1, height, width)
        # images = images.transpose((0, 1, 1, 2))
        # images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(end - start, -1)

    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_activation_statistics_im(ims, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(ims, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(path, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        print(path)
        path = pathlib.Path(path)

        files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.npy'))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def _compute_statistics_im(path, model, batch_size, dims, cuda):



    device=torch.device('cuda')
    model = model.to(device)
    print('load model:', path)
        
    ckpt = torch.load(path, map_location=device)

    # try:
    #     ckpt_name = os.path.basename(apth)
    #     args.start_iter = int(os.path.splitext(ckpt_name)[0])
        
    # except ValueError:
    #     pass
        
    g_ema = Generator(
        256, 512, 8, channel_multiplier=2
    ).to(device)
    g_ema.eval()

    # generator.load_state_dict(ckpt['g'])
    # discriminator.load_state_dict(ckpt['d'])
    g_ema.load_state_dict(ckpt['g_ema'])

    pred_arr = np.empty((356, dims))

    for i in range(0, 356, args.batch_size):
        print(i)
        start = i
        end = min(356, i + args.batch_size)

        noise = mixing_noise(end - start, 512, 0.9, device)

        with torch.no_grad():
            fake_img, _ = g_ema(noise)
        # batch = torch.from_numpy(batch).type(torch.FloatTensor)
        
        fake_img = fake_img.repeat(1, 3, 1, 1)

        pred = model(fake_img)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(-1, dims)


    # # if path.endswith('.npz'):
    # #     f = np.load(path)
    # #     m, s = f['mu'][:], f['sigma'][:]
    # #     f.close()
    # # else:
    # print(path)
    # im = imread(str(path))
    # # im = imread(path.astype(np.float32))
    # # images_3ch = []
    # # for im in images:
    # #     channels = np.stack((im,)*3, axis=-1).reshape(3,im.shape[1], im.shape[0])
    # #     images_3ch.append(channels)
    # # images = np.array(images_3ch)
    # im = im.transpose((2, 0, 1))[np.newaxis, ...]

    # pred_arr = np.empty((16, dims))


    # for i in range(4):
    #     for j in range(4):

    #         # im = imread(str(path))
    #         # im = imread(path.astype(np.float32))
    #         local_im = im[:, :, 2 + 802 * i  : 802 + 802 * i, 2 + 258 * j  : 258 + 258 * j]
    #         print(local_im.shape)
    #         # images_3ch = []
    #         # for im in images:
    #         #     channels = np.stack((im,)*3, axis=-1).reshape(3,im.shape[1], im.shape[0])
    #         #     images_3ch.append(channels)
    #         # images = np.array(images_3ch)
    #         batch = local_im

    #         batch = torch.from_numpy(batch).type(torch.FloatTensor)
    #         if cuda:
    #             batch = batch.cuda()

    #         pred = model(batch)[0]

    #         # If model output is not scalar, apply global spatial average pooling.
    #         # This happens if you choose a dimensionality not equal 2048.
    #         if pred.shape[2] != 1 or pred.shape[3] != 1:
    #             pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    #         pred_arr[i * 4 + j] = pred.cpu().data.numpy().reshape(1, -1)


    m = np.mean(pred_arr, axis=0)
    s = np.cov(pred_arr, rowvar=False)


    # m, s = calculate_activation_statistics(im, model, batch_size,
    #                                        dims, cuda)

    return m, s


# def calculate_fid_given_paths(paths, batch_size, cuda, dims):
#     """Calculates the FID of two paths"""
#     for p in paths:
#         if not os.path.exists(p):
#             raise RuntimeError('Invalid path: %s' % p)

#     block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

#     model = InceptionV3([block_idx])
#     if cuda:
#         model.cuda()

#     m1, s1 = _compute_statistics_of_path(paths[0], model, batch_size,
#                                          dims, cuda)
#     m2, s2 = _compute_statistics_of_path(paths[1], model, batch_size,
#                                          dims, cuda)
#     fid_value = calculate_frechet_distance(m1, s1, m2, s2)

#     return fid_value


if __name__ == '__main__':
    args = parser.parse_args()


    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]

    model = InceptionV3([block_idx])
    if args.gpu:
        model.cuda()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # real
    m1, s1 = _compute_statistics_of_path("../data/wbmri_medium", model, args.batch_size,
                                         args.dims, args.gpu)





    path = pathlib.Path(args.path)
    files = sorted(list(path.glob('*.pt')))[::-1]
    # files = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.npy'))
    


    
    for i in range(len(files)):
        # fake
        m2, s2 = _compute_statistics_im(files[i], model, args.batch_size,
                                             args.dims, args.gpu)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)


        print(files[i], fid_value)

    # fid_value = calculate_fid_given_paths(args.path,
    #                                       args.batch_size,
    #                                       args.gpu != '',
    #                                       args.dims)
    # print('FID: ', fid_value)