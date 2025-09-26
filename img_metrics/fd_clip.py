import os
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from scipy.linalg import sqrtm

def load_clip_model(device):
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess

def image_paths_from_folder(folder):
    exts = ('.png', '.jpg', '.jpeg', '.bmp')
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

def get_clip_features(folder, model, preprocess, device):
    paths = image_paths_from_folder(folder)
    features = []

    for path in tqdm(paths, desc=f"Processing {os.path.basename(folder)}"):
        try:
            image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image).cpu().numpy()
            features.append(feat[0])
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    return np.array(features)

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Computes the Fréchet Distance between two multivariate Gaussians."""
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        print("Adding epsilon to diagonal of covariances.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

def compute_clip_frechet(folder1, folder2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model, preprocess = load_clip_model(device)

    features1 = get_clip_features(folder1, model, preprocess, device)
    features2 = get_clip_features(folder2, model, preprocess, device)

    mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
    mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

if __name__ == "__main__":
    folder_real = "/mnt/Data/musa7216/10k/bdd100k/images/10k/train/"
    folder_fake = "/mnt/Data/musa7216/10k/generations_edge_best/"

    fid_score = compute_clip_frechet(folder_real, folder_fake)
    print(f"CLIP-Frechet Distance: {fid_score:.4f}")