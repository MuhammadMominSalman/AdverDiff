import torch
import os
from huggingface_hub import HfApi
from pathlib import Path
from diffusers.utils import load_image
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
image_segmentor = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")

checkpoint = "lllyasviel/control_v11p_sd15_seg"

image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15_seg/resolve/main/images/input.png"
)

prompt = "old house in stormy weather with rain and wind"

pixel_values = image_processor(image, return_tensors="pt").pixel_values
with torch.no_grad():
  outputs = image_segmentor(pixel_values)
seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
for label, color in enumerate(ada_palette):
    color_seg[seg == label, :] = color
color_seg = color_seg.astype(np.uint8)
control_image = Image.fromarray(color_seg)

control_image.save("./images/control.png")

controlnet = ControlNetModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=30, generator=generator, image=control_image).images[0]

image.save('images/image_out.png')
