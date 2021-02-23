"""AI Summer pathology MICCAI competition 2019"""
import glob
import os

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from dataloader import Gleason2019SaveDISK
from model import Unet
from utils import shuffle_lists

# Data preparation
generate_sub_images = True
root_path = './MICCAI_2019_pathology_challenge/'
folder_to_save_train_samples = './train_samples'
folder_to_save_val_samples = './val_samples'
train_imgs = sorted(glob.glob(os.path.join(root_path, 'Train Imgs/Train Imgs/*.jpg')))

# You have to generate the labels first by running the script
labels_final = sorted(glob.glob('./labels/*.png'))
assert len(labels_final) == len(train_imgs)
train_imgs, labels_final = shuffle_lists(train_imgs, labels_final)
val_loader = Gleason2019SaveDISK('val', train_imgs, labels_final, (0.8, 0.2),
                                 (512, 512), samples=10)
train_loader = Gleason2019SaveDISK('train', train_imgs, labels_final, (0.8, 0.2),
                                   (512, 512), samples=40)
if generate_sub_images:
    val_loader.generate_data(folder_to_save_val_samples)
    train_loader.generate_data(folder_to_save_train_samples)
else:
    print('You have to generate the image samples once')
    train_loader.load_paths()
    val_loader.load_paths()

# Let pytorch lightning handle the training process
in_channels = 3
classes = 7
train_dl = DataLoader(train_loader, batch_size=4, num_workers=8)
val_dl = DataLoader(val_loader, batch_size=4, num_workers=8)
model = Unet(in_channels, classes)
trainer = Trainer(gpus=1, progress_bar_refresh_rate=50, max_epochs=50)
trainer.fit(model, train_dl, val_dl)
