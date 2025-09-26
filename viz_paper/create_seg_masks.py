import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import BDD_100K_Dataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict



# Configs
data_dir = '/mnt/Data/musa7216/BDD-100K/'
resume_path = './models/control_v11p_sd15_seg.pth'
batch_size = 2
accu_grad = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
preprocessor = OneformerCOCODetector()

model = create_model('./models/control_v11p_sd15_seg.yaml').cpu()
model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cpu'), strict=False)
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = BDD_100K_Dataset(data_dir=data_dir, mode="train")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, accumulate_grad_batches=accu_grad, precision=16, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)