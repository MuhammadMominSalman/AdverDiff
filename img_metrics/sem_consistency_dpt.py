import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score
from torchvision.transforms import Resize
from transformers import DPTForSemanticSegmentation, DPTImageProcessor
import pandas as pd
import argparse

# CONFIG VAR
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-real','--realdir', type=str, default='./imgs/ex_dir0')
parser.add_argument('-synth','--synthdir', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o','--out', type=str, default='./sem_con_results.csv')
opt = parser.parse_args()

real_dir = opt.realdir
synthetic_dir = opt.synthdir
image_size = (512, 512)
all_classes = list(range(19))
selected_class_ids = [0, 2, 8, 10, 13]  # road, building, vegetation, sky, car

# GET MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"


model_name = "Intel/dpt-large-ade" 
model = DPTForSemanticSegmentation.from_pretrained("/mnt/Data/musa7216/dpt/dpt-bdd100k/checkpoint-55000", num_labels=19,ignore_mismatched_sizes=True).to(device)

model.eval()
feature_extractor = DPTImageProcessor.from_pretrained(model_name)
resize = Resize(image_size)

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = resize(image)
    encoding = feature_extractor(images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in encoding.items()}

def get_segmentation_mask(image_tensor):
    with torch.no_grad():
        outputs = model(**image_tensor)
        logits = outputs.logits  # shape: (1, num_classes, H, W)
        preds = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    return preds

def compute_class_filtered_jaccard(mask1, mask2, selected_classes):
    mask1 = mask1.flatten()
    mask2 = mask2.flatten()
    selected_mask = np.isin(mask1, selected_classes) | np.isin(mask2, selected_classes)
    mask1 = mask1[selected_mask]
    mask2 = mask2[selected_mask]

    if mask1.size == 0 or mask2.size == 0:
        return 0.0 

    return jaccard_score(mask1, mask2, average="macro", labels=selected_classes)

results = []
image_names = sorted(os.listdir(synthetic_dir))

for img_name in tqdm(image_names, desc="Evaluating 5-class mIoU consistency"):
    synth_path = os.path.join(synthetic_dir, img_name)
    real_img_name = str.split(img_name, sep="_")[-1]
    real_path = os.path.join(real_dir, real_img_name)
    if real_path[-1] == "'":
        real_path = real_path[:-1]

    if not os.path.exists(real_path):
        continue

    real_tensor = preprocess_image(real_path)
    synth_tensor = preprocess_image(synth_path)

    real_mask = get_segmentation_mask(real_tensor)
    synth_mask = get_segmentation_mask(synth_tensor)

    iou_score = compute_class_filtered_jaccard(real_mask, synth_mask, selected_class_ids)

    results.append({
        "image": img_name,
        "5_class_mIoU": iou_score
    })

# Save Results
df = pd.DataFrame(results)
df.to_csv(opt.out, index=False)
print(df.describe())
