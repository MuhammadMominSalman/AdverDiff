# AdverDiff  

**ControlNet‑Based Image Translation**  

A framework for image translation leveraging ControlNet and IP2P diffusion techniques.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Configuration / Parameters](#configuration--parameters)  
- [Training](#training)  
- [Evaluation / Metrics](#evaluation--metrics)
- [Contributing](#contributing)  
- [License](#license)  


## Overview

AdverDiff is a system for translating or transforming images using ControlNet-based architectures with adversarial diffusion. The goal is to produce high-quality, controllable image translations guided by structural cues, style inputs, or domain constraints.


## Project Structure

Here is an outline of the main folders and files in the repository:  

```
AdverDiff/
├── train/              — Training scripts  
├── test/               — Inference / testing scripts  
├── datasets_folder/    — Datasets, preprocessing, loaders  
├── ldm/                — Latent diffusion modules  
├── cldm/               — ControlNet / conditional modules  
├── img_metrics/        — Image quality / evaluation metrics  
├── viz_paper/          — Visualization and figure scripts for paper  
├── prework/            — Preprocessing / auxiliary code  
├── gradios/            — Gradios to make an web app  
├── .gitignore  
├── LICENSE  
├── environment.yaml  
├── requirements.txt  
└── README.md  
```  


## Installation

**Setup**

1. Clone the repo:

   ```bash
   git clone https://github.com/MuhammadMominSalman/AdverDiff.git
   cd AdverDiff
   ```

2. (Optional) Create a conda environment:

   ```bash
   conda env create -f environment.yaml
   conda activate adverdiff
   ```

   Or use `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure all dependencies are installed (PyTorch, CUDA toolkit, torchvision, etc.).  


## Usage

Here’s how you run the main functions (training, testing). These are hypothetical commands — adjust them to your actual scripts.

### Training

```bash
python train/train_**.py
```

### Testing / Inference

```bash
python test/inference.py --input_dir path/to/input --output_dir path/to/output --ckpt path/to/checkpoint.pth
```

You can also integrate with scripts in `test/` and visualize via `viz_paper/`.

## Configuration / Parameters

You can control model behavior via configuration files (YAML) or command‑line arguments. Some common parameters include:

- `--lr`: learning rate  
- `--batch_size`: batch size  
- `--num_epochs`: number of training epochs  
- `--ckpt_path`: path to checkpoint  
- `--input_size`, `--image_size`: input / output image dimensions  
- ControlNet-related weights, loss coefficients, etc.  
- Paths: dataset path, output directories  

Provide a sample configuration or default config file (e.g. `configs/default.yaml`) to accompany the code.

## Training

1. Prepare your dataset.
2. Configure the YAML or argument file with dataset paths, hyperparameters.  
3. Run the training script.  
4. Monitor training losses, checkpoint models checkpoints in `train/` or designated folder.  
5. Optionally resume from checkpoints or fine-tune.

## Evaluation / Metrics

- The `img_metrics/` directory contains routines for computing image quality metrics such as FID, PSNR, SSIM, LPIPS, etc.  
- Use these scripts to compare generated images vs. ground truth or baselines.  
- Visualization scripts in `viz_paper/` can help generate figures for reports or publications.


## Contributing

Contributions are welcome!  

## License

This project is licensed under the **MIT License**.  
Please see the [LICENSE](LICENSE) file for details.  


## Acknowledgments / References

- This repository is forked and built upon [ControlNet v1.1 Nightly](https://github.com/lllyasviel/ControlNet-v1-1-nightly/tree/main)  