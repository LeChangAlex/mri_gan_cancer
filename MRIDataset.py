import argparse
import os
import numpy as np
import math
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from collections import OrderedDict

class MRIDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.annotations = pd.read_csv(csv_file, engine='python')
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return (len(self.annotations))

    def __getitem__(self,index):
        volume_name = os.path.join(self.root_dir,
        self.annotations.iloc[index,0])
        np_volume = np.load(volume_name)
        volume = Image.fromarray(np_volume)
        # annotations = self.annotations.iloc[index,0].as_matrix()
        # annotations = annotations.astype('float').reshape(-1,2)
        sample = volume

        if self.transform:
            sample = self.transform(sample)
        
        return sample


