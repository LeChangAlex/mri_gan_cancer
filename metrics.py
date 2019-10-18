import sklearn.mixture
import numpy as np
import scipy as sp
from MRIDataset import *
import torch.nn as nn

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


    def calculate_fd(self, gen_data):
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

        diff = self.real_mean - gen_mean
        a = self.real_cov.dot(gen_cov, out=gen_cov)

        fd = diff.dot(diff) + np.trace(self.real_cov + gen_cov - 2 * (a ** 0.5))

        return fd
