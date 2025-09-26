from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from annotator.canny import CannyDetector
from cldm.ddim_hacked import DDIMSampler
import pandas as pd
import os
from PIL import Image
import os

data_dir = '/mnt/Data/musa7216/10k/'


detect_resolution = 512
image_resolution = 512
ddim_steps = 20
num_samples = 2
seed = 1
strength = 1
eta = 1
scale = 9
guess_mode = False



directory = os.path.join(data_dir, "train")
for _, _, filenames in os.walk(directory):
    for filename in filenames:
        if not os.path.isfile(os.path.join(data_dir, "train_image_resized", "{f}".format(f=filename))):
            input_image = cv2.imread(os.path.join(data_dir, "train", filename))
            # Do not forget that OpenCV read images in BGR order.
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

            # Resize the images to 512x512
            input_image = cv2.resize(input_image, (512, 512), interpolation = cv2.INTER_LINEAR)
            if not os.path.isdir(os.path.join(data_dir, "train_image_resized")):
                os.makedirs(os.path.join(data_dir, "train_image_resized"))
            im = Image.fromarray(input_image)
            im.save(os.path.join(data_dir, "train_image_resized", "{f}".format(f=filename)))