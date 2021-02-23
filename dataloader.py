import os

import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle
from utils import make_dirs


class Gleason2019(Dataset):
    """
    Code for reading Gleason 2019 MICCAI Challenge
    """

    def __init__(self, mode, image_paths, label_paths, split=(0.8, 0.2), crop_dim=(512, 512), samples=100):
        """
        :param mode: 'train','val'
        :param image_paths: image dataset paths
        :param label_paths: label dataset paths
        :param crop_dim: 2 element tuple to decide crop values
        :param samples: number of sub-grids to create(patches of the input img)
        """
        self.slices = 244
        self.mode = mode
        self.crop_dim = crop_dim
        self.sample_list = []
        self.samples = samples
        train_idx = int(split[0] * self.slices)
        val_idx = int(split[1] * self.slices)

        if self.mode == 'train':
            self.list_imgs = image_paths[0:train_idx]
            self.list_labels = label_paths[0:train_idx]
            self.generate_samples()
        elif self.mode == 'val':
            self.list_imgs = image_paths[train_idx:(train_idx + val_idx)]
            self.list_labels = label_paths[train_idx:(train_idx + val_idx)]
            self.generate_samples()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        tuple_in = self.sample_list[index]
        img_tensor, segmentation_map = tuple_in
        return img_tensor, segmentation_map

    def generate_samples(self):
        total = len(self.list_imgs)
        print('Total ' + self.mode + ' data to generate samples:', total)
        for j in range(total):
            for i in range(self.samples):
                input_path = self.list_imgs[j]
                label_path = self.list_labels[j]

                img_numpy = imageio.imread(input_path)
                label_numpy = imageio.imread(label_path)

                img_numpy, label_numpy = self.crop_img(img_numpy, label_numpy)

                img_tensor = torch.from_numpy(img_numpy).float()
                label_tensor = torch.from_numpy(label_numpy).unsqueeze(0)

                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = norm_img(img_tensor)
                self.sample_list.append(tuple((img_tensor, label_tensor)))

    def generate_patch(self, img):
        h, w, c = img.shape
        if h < self.crop_dim[0] or w < self.crop_dim[1]:
            print('dim error')
            print(h, self.crop_dim[0], w, self.crop_dim[1])
        h_crop = np.random.randint(h - self.crop_dim[0])
        w_crop = np.random.randint(w - self.crop_dim[1])
        return h_crop, w_crop

    def crop_img(self, img_numpy, label_numpy):
        h_crop, w_crop = self.generate_patch(img_numpy)
        img_numpy = img_numpy[h_crop:(h_crop + self.crop_dim[0]),
                    w_crop:(w_crop + self.crop_dim[1]), :]
        label_numpy = label_numpy[h_crop:(h_crop + self.crop_dim[0]),
                      w_crop:(w_crop + self.crop_dim[1])]
        return img_numpy, label_numpy


class Gleason2019SaveDISK(Dataset):
    """
    Code for reading Gleason 2019 MICCAI Challenge
    """

    def __init__(self, mode, image_paths, label_paths, split=(0.8, 0.2), crop_dim=(512, 512), samples=100):
        """
        :param mode: 'train','val'
        :param image_paths: image dataset paths
        :param label_paths: label dataset paths
        :param crop_dim: 2 element tuple to decide crop values
        :param samples: number of sub-grids to create(patches of the input img)
        """
        self.slices = 244
        self.mode = mode
        self.crop_dim = crop_dim
        self.sample_list = []
        self.samples = samples
        self.train_idx = int(split[0] * self.slices)
        self.val_idx = int(split[1] * self.slices)
        self.image_paths = image_paths
        self.label_paths = label_paths

    def generate_data(self, path):
        make_dirs(path)
        sample_img_path = os.path.join(path, 'sample_imgs')
        sample_seg_path = os.path.join(path, 'sample_seg')

        make_dirs(sample_img_path)
        make_dirs(sample_seg_path)

        if self.mode == 'train':
            self.list_imgs = self.image_paths[0:self.train_idx]
            self.list_labels = self.label_paths[0:self.train_idx]
            self._generate_samples(sample_img_path, sample_seg_path)
        elif self.mode == 'val':
            self.list_imgs = self.image_paths[self.train_idx:(self.train_idx + self.val_idx)]
            self.list_labels = self.label_paths[self.train_idx:(self.train_idx + self.val_idx)]
            self._generate_samples(sample_img_path, sample_seg_path)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        out_img_file, out_seg_file = self.sample_list[index]
        img_tensor = torch.from_numpy(np.load(out_img_file))
        segmentation_map = torch.from_numpy(np.load(out_seg_file))
        return img_tensor, segmentation_map

    def _generate_samples(self, sample_img_path, sample_seg_path):
        total = len(self.list_imgs)
        print('Total ' + self.mode + ' data to generate samples:', total)
        for j in range(total):
            for i in range(self.samples):
                input_path = self.list_imgs[j]
                label_path = self.list_labels[j]

                img_numpy = imageio.imread(input_path)
                label_numpy = imageio.imread(label_path)

                img_numpy, label_numpy = self.crop_img(img_numpy, label_numpy)

                img_tensor = torch.from_numpy(img_numpy).float()
                label_tensor = torch.from_numpy(label_numpy).unsqueeze(0)
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = norm_img(img_tensor)

                img_name = f"{self.mode}_img_{str(j).zfill(3)}_sample_{str(i).zfill(3)}.npy"
                out_img_file = os.path.join(sample_img_path, img_name)

                seg_name = f"{self.mode}_seg_{str(j).zfill(3)}_sample_{str(i).zfill(3)}.npy"
                out_seg_file = os.path.join(sample_seg_path, seg_name)

                np.save(out_img_file, img_tensor.numpy())
                np.save(out_seg_file, label_tensor.numpy())
                self.sample_list.append(tuple((out_img_file, out_seg_file)))
        # save
        with open(f'sub_img_paths_{self.mode}', 'wb') as fp:
            pickle.dump(self.sample_list, fp)

    def load_paths(self, ):
        with open(f'sub_img_paths_{self.mode}', 'rb') as fp:
            self.sample_list = pickle.load(fp)

    def generate_patch(self, img):
        h, w, c = img.shape
        if h < self.crop_dim[0] or w < self.crop_dim[1]:
            print('dim error')
            print(h, self.crop_dim[0], w, self.crop_dim[1])
        h_crop = np.random.randint(h - self.crop_dim[0])
        w_crop = np.random.randint(w - self.crop_dim[1])
        return h_crop, w_crop

    def crop_img(self, img_numpy, label_numpy):
        h_crop, w_crop = self.generate_patch(img_numpy)
        img_numpy = img_numpy[h_crop:(h_crop + self.crop_dim[0]),
                    w_crop:(w_crop + self.crop_dim[1]), :]
        label_numpy = label_numpy[h_crop:(h_crop + self.crop_dim[0]),
                      w_crop:(w_crop + self.crop_dim[1])]
        return img_numpy, label_numpy


def norm_img(img_tensor):
    mask = img_tensor.ne(0.0)
    desired = img_tensor[mask]
    mean_val, std_val = desired.mean(), desired.std()
    img_tensor = (img_tensor - mean_val) / std_val
    return img_tensor
