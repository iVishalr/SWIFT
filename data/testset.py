import os
import torch
from typing import Tuple
from heic2png import HEIC2PNG
from torch.utils.data import Dataset
import data.utils_image as img_utils

class TestSet(Dataset):
    def __init__(self, hr_path, lr_path, scale, testing=False) -> None:
        self.hr_path = hr_path
        self.lr_path = lr_path
        self.scale = scale
        self.testing = testing

        for img_path in os.listdir(self.hr_path):
            ext = img_path.split(".")[-1]
            if ext == "HEIC" or ext == "heic":
                heic_img = HEIC2PNG(self.hr_path + img_path)
                os.remove(self.hr_path + img_path)
                heic_img.save()

        self.hr_images = sorted([os.path.join(self.hr_path, img_path) for img_path in os.listdir(self.hr_path) if img_path != ".DS_Store"])

        if self.lr_path != None:
            self.lr_images = sorted([os.path.join(self.lr_path, img_path) for img_path in os.listdir(self.lr_path)])
            assert len(self.hr_images) == len(self.lr_images), "Number of HR Images does not match Number of LR Images"
        else:
            self.lr_images = None

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.lr_images != None:
            hr_image = img_utils.imread_uint(self.hr_images[index])
            hr_image = img_utils.uint2single(hr_image)
            hr_image = img_utils.modcrop(hr_image, self.scale)
            lr_image = img_utils.imread_uint(self.lr_images[index])
            lr_image = img_utils.uint2single(lr_image)
        else:
            hr_image = img_utils.imread_uint(self.hr_images[index])
            hr_image = img_utils.uint2single(hr_image)
            hr_image = img_utils.modcrop(hr_image, self.scale)
            H, W = hr_image.shape[0], hr_image.shape[1]
            lr_image = img_utils.imresize_np(hr_image, 1 / self.scale, True)
        
        lr_tensor, hr_tensor = img_utils.single2tensor3(lr_image), img_utils.single2tensor3(hr_image)
        
        if self.testing:
            return {"lr": hr_tensor, "hr": hr_tensor}
            
        return {"lr": lr_tensor, "hr": hr_tensor, "lr_path": -1 if self.lr_path is None else self.lr_images[index], "hr_path": self.hr_images[index]}