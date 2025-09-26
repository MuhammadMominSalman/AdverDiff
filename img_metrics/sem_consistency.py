import os
import sys
sys.path.insert(1, '/home/musa7216/ControlNet-v1-1-nightly/')
import argparse
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from annotator.util import resize_image, HWC3
from annotator.oneformer import OneformerADE20kDetector
from sklearn.metrics import jaccard_score
from tqdm import tqdm
from scipy.spatial.distance import cosine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
seg_model = OneformerADE20kDetector()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


def load_and_preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    img_t = np.array(img)
    return img_t, img

def compute_clip_score(img1, img2):
    inputs = clip_processor(images=[img1, img2], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    img1_feat, img2_feat = outputs[0], outputs[1]
    img1_feat /= img1_feat.norm(dim=-1, keepdim=True)
    img2_feat /= img2_feat.norm(dim=-1, keepdim=True)
    return 1 - cosine(img1_feat.cpu().numpy(), img2_feat.cpu().numpy())


def evaluate_semantic_consistency(real_dir, synth_dir):
    synth_files = [os.path.join(synth_dir, f) for f in os.listdir(synth_dir) if f.endswith(('.png', '.jpg'))]
    real_files = [os.path.join(real_dir, f.split('_')[-1]) for f in os.listdir(synth_dir) if f.endswith(('.png', '.jpg'))]
    clip_scores = []

    for real_path, synth_path in tqdm(zip(real_files, synth_files), total=len(real_files)):
        real_pil = Image.open(real_path).convert("RGB")
        synth_pil = Image.open(synth_path).convert("RGB")

        # CLIP Score
        clip = compute_clip_score(real_pil, synth_pil)
        clip_scores.append(clip)

    return {
        "Average CLIPScore": np.mean(clip_scores)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d0','--dir0', type=str, default='./imgs/ex_dir0')
    parser.add_argument('-d1','--dir1', type=str, default='./imgs/ex_dir1')
    parser.add_argument('-o','--out', type=str, default='./imgs/example_dists.txt')

    opt = parser.parse_args()
    folder_real = opt.dir0
    folder_fake = opt.dir1
    results = evaluate_semantic_consistency(folder_real, folder_fake)
    f = open(opt.out,'w')
    f.writelines(results)
    f.close()
    print("\n <<Semantic Perceptual Evaluation Results>>")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
