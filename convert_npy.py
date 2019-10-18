import argparse
import os
import numpy as np
import math
import pandas as pd
from PIL import Image
import glob

real_image_folder = './wbrmi_slices_medium/'
real_png_folder = './wbmri_slices_medium_png/'

for file in glob.glob('./wbmri_slices_medium/*.npy'):
    print(np.load(file))
    im_array = np.load(file)
    im_array_scaled = (((im_array - im_array.min()) / (im_array.max() - im_array.min())) * 255.9).astype(np.uint8)
    im = Image.fromarray(im_array_scaled)
    im_name = file.split('.npy')[0].split('/')[2] + ".png"
    print(im_name)
    im.save(real_png_folder + im_name)