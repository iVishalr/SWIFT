
import os
import cv2
import random
import numpy as np
import torch
import albumentations as A
import torch.utils.data as data

from tqdm import tqdm
from data import common
from typing import Tuple
from torch.utils.data import DataLoader

def default_loader(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]
    return img

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class Dataset(data.Dataset):
    def __init__(self, opt, load_dataset=True):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext   # '.png' (default) or '.npy'
        self.train = True if self.opt.phase == 'train' else False
        self.repeat = 1 #self.opt.test_every // (self.opt.n_train // self.opt.batch_size)
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()
        self.load_dataset = load_dataset

        self.transforms = A.Compose([
                A.ChannelShuffle(p=0.35)
            ],
            additional_targets={"LR": 'image'}
        )

        if load_dataset:
            self.images_hr_data, self.images_lr_data = self.images_hr.copy(), self.images_hr.copy()

            for i in tqdm(range(len(self.images_hr)), total=len(self.images_hr)):
                idx = self._get_index(i)
                if self.ext == '.npy':
                    lr = npy_loader(self.images_lr[idx])
                    hr = npy_loader(self.images_hr[idx])
                else:
                    lr = default_loader(self.images_lr[idx])
                    hr = default_loader(self.images_hr[idx])
                self.images_lr_data[i] = lr
                self.images_hr_data[i] = hr

    def _set_filesystem(self, dir_data):
        self.root = dir_data + '/DIV2K'
        self.dir_hr = os.path.join(self.root, 'DIV2K_train_HR/HR')
        self.dir_lr = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X' + str(self.scale))

    def __getitem__(self, idx):
        if self.load_dataset:
            idx = self._get_index(idx)
            lr, hr = self.images_lr_data[idx], self.images_hr_data[idx]
        else:
            lr, hr = self._load_file(idx)

        lr, hr = self._get_patch(lr, hr)
        
        mode = np.random.randint(0,8)
        lr = self.augment_img(lr, mode)
        hr = self.augment_img(hr, mode)

        transformed = self.transforms(image=hr, LR=lr)
        hr, lr = transformed["image"], transformed["LR"]

        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr_tensor, hr_tensor = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self):
        if self.train:
            return self.opt.n_train * self.repeat

    def _get_index(self, idx):
        if self.train:
            return idx % self.opt.n_train
        else:
            return idx

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)

        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        idx = self._get_index(idx)
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr

    def get_random_masks(self, p, image_size: Tuple, patch_size: int, num_masks=3):
        mask = None
        if random.random() >= p:
            H,W,C = image_size
            random.seed(random.randint(patch_size, H - patch_size))

            mask = np.ones((H,W,C))

            # get random numbers within a specific range
            random_start_points = set()
            while len(random_start_points) < num_masks:
                random_start_points.add(random.randrange(patch_size, H - patch_size, patch_size))

            for points in random_start_points:
                h,w = (points,points)
                mask[h:h+patch_size,w:w+patch_size,:] = 0     
        return mask

    def augment_img(self, img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return np.flipud(np.rot90(img))
        elif mode == 2:
            return np.flipud(img)
        elif mode == 3:
            return np.rot90(img, k=3)
        elif mode == 4:
            return np.flipud(np.rot90(img, k=2))
        elif mode == 5:
            return np.rot90(img)
        elif mode == 6:
            return np.rot90(img, k=2)
        elif mode == 7:
            return np.flipud(np.rot90(img, k=3))


    def augment_img_tensor4(self, img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        if mode == 0:
            return img
        elif mode == 1:
            return img.rot90(1, [2, 3]).flip([2])
        elif mode == 2:
            return img.flip([2])
        elif mode == 3:
            return img.rot90(3, [2, 3])
        elif mode == 4:
            return img.rot90(2, [2, 3]).flip([2])
        elif mode == 5:
            return img.rot90(1, [2, 3])
        elif mode == 6:
            return img.rot90(2, [2, 3])
        elif mode == 7:
            return img.rot90(3, [2, 3]).flip([2])


    def augment_img_tensor(self, img, mode=0):
        '''Kai Zhang (github: https://github.com/cszn)
        '''
        img_size = img.size()
        img_np = img.data.cpu().numpy()
        if len(img_size) == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
        elif len(img_size) == 4:
            img_np = np.transpose(img_np, (2, 3, 1, 0))
        img_np = self.augment_img(img_np, mode=mode)
        img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
        if len(img_size) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        elif len(img_size) == 4:
            img_tensor = img_tensor.permute(3, 2, 0, 1)

        return img_tensor.type_as(img)


    def augment_img_np3(self, img, mode=0):
        if mode == 0:
            return img
        elif mode == 1:
            return img.transpose(1, 0, 2)
        elif mode == 2:
            return img[::-1, :, :]
        elif mode == 3:
            img = img[::-1, :, :]
            img = img.transpose(1, 0, 2)
            return img
        elif mode == 4:
            return img[:, ::-1, :]
        elif mode == 5:
            img = img[:, ::-1, :]
            img = img.transpose(1, 0, 2)
            return img
        elif mode == 6:
            img = img[:, ::-1, :]
            img = img[::-1, :, :]
            return img
        elif mode == 7:
            img = img[:, ::-1, :]
            img = img[::-1, :, :]
            img = img.transpose(1, 0, 2)
            return img


    def augment_imgs(self, img_list, hflip=True, rot=True):
        # horizontal flip OR rotate
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[::-1, :, :]
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return [_augment(img) for img in img_list]

class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)