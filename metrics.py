import sklearn.mixture
import numpy as np
import scipy as sp
from MRIDataset import *
import torch.nn as nn
from scipy import linalg

class DomainFD:
    def __init__(self, ae, ae_resolution, device):
        self.ae = ae
        self.device = device
        self.ae_resolution = ae_resolution
        # self.n_components = n_components
        # self.model = sklearn.mixture.GaussianMixture(n_components=self.n_components)

    # GMM and VAE too time consuming?
    # def fit_gmm(self, data):
    #     self.model.fit(data)

    def fit_real_data(self, data):
        """
        :param data: data loader
        """
        data = next(data).to(self.device)
        with torch.no_grad():
            latent = self.ae.encode(data)

        latent = torch.flatten(latent, start_dim=1).cpu().numpy()

        self.real_mean = np.mean(latent, axis=0)
        self.real_cov = np.cov(latent, rowvar=False)


    def calculate_fd(self, gen_data, eps=1e-6):
        # gen_gmm = sklearn.mixture.GaussianMixture(n_components=self.n_components)
        # gen_gmm.fit(gen_data)
        """
        :param data: tensor
        """
        gen_data = nn.functional.interpolate(gen_data, size=self.ae_resolution,
                                  mode='bilinear', align_corners=False)
        with torch.no_grad():
            latent = self.ae.encode(gen_data)

        latent = torch.flatten(latent, start_dim=1).cpu().numpy()

        gen_mean = np.mean(latent, axis=0)
        gen_cov = np.cov(latent, rowvar=False)

        # diff = self.real_mean - gen_mean
        # a = self.real_cov.dot(gen_cov)
        # b = np.trace(self.real_cov + gen_cov - 2 * (a ** 0.5))
        # fd = diff.dot(diff) + np.trace(self.real_cov + gen_cov - 2 * (a ** 0.5))

        # --------
        mu1 = self.real_mean
        mu2 = gen_mean

        sigma1 = self.real_cov
        sigma2 = gen_cov

        print(mu1, mu2)
        print(sigma1, sigma2)

        diff = mu1 - mu2

        # Product might be almost singular
        # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        covmean = sigma1.dot(sigma2) ** 0.5

        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            # if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            #     m = np.max(np.abs(covmean.imag))
            #     raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        fd = (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
        print(fd)
        return fd

