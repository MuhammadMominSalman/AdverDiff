import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets_folder.dataset import BDD_10K_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# CONFIG VAR
data_dir = '/mnt/Data/musa7216/BDD-100K/'
resume_path = '/mnt/Data/musa7216/cnet_results/models/control_v11p_sd15_seg.pth'
sd_path = '/mnt/Data/musa7216/cnet_results/models/v1-5-pruned.ckpt'
cfg_path = '/mnt/Data/musa7216/cnet_results/models/control_v11p_sd15_seg.yaml'
batch_size = 4
accu_grad = 16
logger_freq = 4000
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(cfg_path).cpu()
model.load_state_dict(load_state_dict(sd_path, location='cpu'), strict=False)
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = BDD_10K_Dataset(data_dir=data_dir, mode="train")
dataset_val = BDD_10K_Dataset(data_dir=data_dir, mode="val")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, num_workers=0, batch_size=batch_size, shuffle=False)


logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(accelerator="gpu", devices=[1], accumulate_grad_batches=accu_grad, 
                     precision=16, check_val_every_n_epoch=5, callbacks=[logger])


# Train!
trainer.fit(model, dataloader, dataloader_val)