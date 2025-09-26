import sys

sys.path.append("/home/musa7216/ControlNet-v1-1-nightly")

from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import pandas as pd
import os
from PIL import Image
import shutil

def copy_and_rename(src_path, dest_path, old_name, new_name):
	# Copy the file
	shutil.copy(src_path, dest_path)

	# Rename the copied file
	new_path = f"{dest_path}/{new_name}"
	shutil.move(f"{dest_path}/{old_name}", new_path)


preprocessor = None

model_name = 'control_v11e_sd15_ip2p'
res_dir = '/mnt/Data/musa7216/cnet_results'
mnt_dir = "/mnt/Data/musa7216/archive_lightning_logs"
model = create_model(os.path.join(res_dir, f'models/{model_name}.yaml')).cpu()
model.load_state_dict(load_state_dict(os.path.join(res_dir, f'models/v1-5-pruned.ckpt'), location='cuda'), strict=False)
model.load_state_dict(load_state_dict(os.path.join(res_dir, f'models/{model_name}.pth'), location='cuda'), strict=False)
model.load_state_dict(load_state_dict(os.path.join(mnt_dir, 'version_2/checkpoints/epoch=6-step=3149.ckpt'),
                                                   location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)

#CONFIG
det = 'Seg_OFCOCO'
data_dir = '/mnt/Data/musa7216/10k/'
scr_dir = '/mnt/Data/musa7216/10k/bdd100k/images/10k/'


detect_resolution = 512
image_resolution = 512
ddim_steps = 20
num_samples = 1
seed = 1
strength = 1
eta = 1
scale = 9
guess_mode = False


df = pd.read_csv(os.path.join(data_dir, "labels", "train_df_10k.csv"))
tod_list = ["day", "night", "dawn"]
weather_list = ["overcast", "partly cloudy", "foggy", "snowy", "rainy"]
while True:
    index = random.sample(range(len(df)),1)[0]
    weather_real = df.iloc[index]["weather"]
    tod_real = df.iloc[index]["timeofday"]
    filename = df.iloc[index]["name"]
    possible_choices_t = [v for v in tod_list if v != tod_real]
    possible_choices_w = [v for v in weather_list if v != weather_real]
    w = random.sample(possible_choices_w,1)[0]
    tod = random.sample(possible_choices_t,1)[0]

    prompt = "A {w} image showing a car driving down a road on a {w} {t}.".format(w=w, t=tod)
    if weather_real != w and tod_real != tod and not os.path.isfile(os.path.join(data_dir, "generations_ip2p_best_2", "{w}_{t}_ip2p_{f}".format(f=filename, w=w, t=tod))):
        input_image = cv2.imread(os.path.join(scr_dir, "train", filename))
        # Do not forget that OpenCV read images in BGR order.
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


        ##Copy GT Segmentation Maps
        source_file = "/mnt/Data/musa7216/10k/bdd100k/labels/sem_seg/masks/train/{f}.png".format(f=filename[:-4])
        destination_folder = "/mnt/Data/musa7216/10k/ip2p_best_gtseg_2/"
        old_file_name = "{f}.png".format(f=filename[:-4])
        new_file_name = "{w}_{t}_edge_{f}.png".format(f=filename[:-4], t=tod,w=w) 
        copy_and_rename(source_file, destination_folder, old_file_name, new_file_name)

        # Resize the images to 512x512
        input_image = cv2.resize(input_image, (512, 512), interpolation = cv2.INTER_LINEAR)

        with torch.no_grad():
            input_image = HWC3(input_image)
            detected_map = input_image.copy()

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                            shape, cond, verbose=False, eta=eta,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            if not os.path.isdir(os.path.join(data_dir, "generations_ip2p_best_2")):
                os.makedirs(os.path.join(data_dir, "generations_ip2p_best_2"))
            for i in range(num_samples):
                im = Image.fromarray(x_samples[i])
                im.save(os.path.join(data_dir, "generations_ip2p_best_2", "{w}_{t}_ip2p_{f}".format(f=filename, w=w, t=tod)))
                    


