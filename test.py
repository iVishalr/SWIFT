import os, gc
import random
import argparse
import datetime

import copy
import time
import utils
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import skimage.color as sc

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.SWIFT import SWIFT
from data import prepare_testset

from util.calculate_psnr_ssim import calculate_psnr, calculate_ssim
from fvcore.nn import FlopCountAnalysis, flop_count_table


parser = argparse.ArgumentParser(description="TESTING")
parser.add_argument("--test_batch_size", type=int, default=1,
                    help="testing batch size")
parser.add_argument("--scale", type=int, default=4,
                    help="scale")
parser.add_argument("--patch_size", type=int, default=256,
                    help="patch size")
parser.add_argument("--model_path", type=str, required=True,
                    help="path to trained model")
parser.add_argument("--cuda", action="store_true", default=True,
                    help="use cuda")
parser.add_argument("--forward_chop", action="store_true", default=False,
                    help="use forward chop")
parser.add_argument("--seed", type=int, default=3407)


args = parser.parse_args()

seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)

random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_path = "./testsets/"

testset_BSDS100 = prepare_testset.MakeTestSet(f"{dataset_path}BSD100/HR", None, args.scale, args.patch_size)
testset_General100 = prepare_testset.MakeTestSet(f"{dataset_path}General100/HR", None, args.scale, args.patch_size)
testset_Manga109 = prepare_testset.MakeTestSet(f"{dataset_path}Manga109/HR", None, args.scale, args.patch_size)
testset_Urban100 = prepare_testset.MakeTestSet(f"{dataset_path}Urban100/HR", None, args.scale, args.patch_size)
testset_Set14 = prepare_testset.MakeTestSet(f"{dataset_path}Set14/HR", None, args.scale, args.patch_size)
testset_Set5 = prepare_testset.MakeTestSet(f"{dataset_path}Set5/HR/", f"{dataset_path}Set5/LR/X{args.scale}/", args.scale, args.patch_size)

BSDS100_data_loader = DataLoader(dataset=testset_BSDS100, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
General100_data_loader = DataLoader(dataset=testset_General100, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
Manga109_data_loader = DataLoader(dataset=testset_Manga109, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
Urban100_data_loader = DataLoader(dataset=testset_Urban100, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
Set14_data_loader = DataLoader(dataset=testset_Set14, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
Set5_data_loader = DataLoader(dataset=testset_Set5, num_workers=0, batch_size=args.test_batch_size, shuffle=False)
test_data_loader_dict = {"Set5": Set5_data_loader, "Set14": Set14_data_loader, "BSD100": BSDS100_data_loader, "Urban100": Urban100_data_loader, "Manga109": Manga109_data_loader, "General100": General100_data_loader}


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    
    input_tensor = torch.randn(1,3,64,64, device=device)
    
    flops = FlopCountAnalysis(net, input_tensor)
    table = flop_count_table(flops)
    print("\nFLOP Analysis Table")
    print("-" * len(table.split("\n")[0]))
    print(table)
    print("-" * len(table.split("\n")[0]))
    print()

    print('\nTotal Number of FLOPs: {:.2f} G'.format(flops.total() / 1e9))
    print('\nTotal number of parameters: %d\n' % num_params)

def forward_chop(model, x, scale, shave=10, min_size=60000):
    n_GPUs = 1 #min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, scale, shave=shave, min_size=min_size) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def test(model_path, model_type="small"):

    if model_type == 'small':
        base_model = SWIFT(
            img_size=args.patch_size//args.scale,
            patch_size=1,
            in_channels=3,
            embd_dim=64,
            rfbs=[2, 2, 2, 2],
            depths=[2, 2, 2, 2],
            num_heads=[8, 8, 8, 8],
            mlp_ratio=1,
            window_size=8,
            residual_conv="3conv",
            scale=args.scale,
            act_layer=nn.GELU,
            feat_scale=False,
            attn_scale=True,
        )
    elif model_type == 'medium':
        base_model = SWIFT(
            img_size=args.patch_size//args.scale,
            patch_size=1,
            in_channels=3,
            embd_dim=64,
            rfbs=[2, 2, 2, 2, 2],
            depths=[2, 2, 2, 2, 2],
            num_heads=[8, 8, 8, 8, 8],
            mlp_ratio=1,
            window_size=8,
            residual_conv="3conv",
            scale=args.scale,
            act_layer=nn.GELU,
            feat_scale=False,
            attn_scale=True,
        )
    
    model_paths = [
        (args.model_path if args.model_path else "/home/vishalr/Desktop/SWIFT/experiment/SWIFT-S-4x-662k.pth", 1), #(path, weight)
    ]

    models = []

    for model_path, model_weights in model_paths:
        model = copy.deepcopy(base_model)
        print("loading :", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=True)

        model.to(device)
        model.eval()
        models.append((model, model_weights))

    print("Starting to Test")
    for testset, test_data_loader in test_data_loader_dict.items():
        test_iter = 1
        avg_psnr_y, avg_ssim_y = 0, 0

        start = []
        end = []
        print(f"Testset : {testset}")
        if not testset == "Set5" and not testset == "Personal":

            if not os.path.exists(os.path.join(dataset_path, testset, "LR", f"X{args.scale}")):
                os.makedirs(os.path.join(dataset_path, testset, "LR", f"X{args.scale}"))
            
            lr_save_path = os.path.join(dataset_path, testset, "LR", f"X{args.scale}")

            # check if we actually need to save the files.
            num_lr_imgs = len(sorted([x for x in os.listdir(lr_save_path) if x != ".DS_Store"]))
            num_hr_imgs = len(sorted([x for x in os.listdir(os.path.join(dataset_path, testset, "HR")) if x != ".DS_Store"]))

            print(f"Number of HR Images : {num_hr_imgs}")
            print(f"Number of LR Images : {num_lr_imgs}")

            if not num_hr_imgs == num_lr_imgs:
                print(f"Dataset Missing! Generating LR images for X{args.scale} scale.")
                for i, batch in enumerate(test_data_loader):
                    lr_tensor, lr_path = batch["lr"], batch["lr_path"]
                    hr_path = batch["hr_path"][0].split("/")[-1]
                    if lr_path == -1:
                        lr_img = utils.tensor2np(lr_tensor.detach()[0])
                        lr_img = Image.fromarray(lr_img)
                        lr_img.save(os.path.join(lr_save_path, hr_path))
                        print(f"Saving {os.path.join(lr_save_path, hr_path)}")

        for i, batch in enumerate(test_data_loader):
            lr_tensor, hr_tensor = batch["lr"], batch["hr"]
            _,_, h_old, w_old = lr_tensor.size()

            if args.cuda:
                lr_tensor = lr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)

            with torch.no_grad():
                _, _, h_old, w_old = lr_tensor.size()
                h_pad = (h_old // model.window_size + 1) * model.window_size - h_old
                w_pad = (w_old // model.window_size + 1) * model.window_size - w_old
                lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
                lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

                if args.forward_chop:
                    start.append(time.time())
                    pre = forward_chop(model, lr_tensor, args.scale) # saves memory on during testing on very large images
                    end.append(time.time()-start[-1])
                else:
                    start.append(time.time())
                    pre = None
                    pred = []
                    for model,w in models:
                        pre = model(lr_tensor)
                        pre = pre[..., :h_old * args.scale, :w_old * args.scale]
                        pred.append((pre,w))
                    end.append(time.time()-start[-1])

            for pre,w in pred:
                sr_img = utils.tensor2np(pre.detach()[0])
                gt_img = utils.tensor2np(hr_tensor.detach()[0])

                if testset == 'Personal':
                    print(sr_img.shape)
                    output_img = Image.fromarray(sr_img)
                    output_img.save(f"./results/SWIFTx{args.scale}/{i+1}.png")
                    continue

                if sr_img.ndim == 3: 
                    sr_img = sr_img[:,:,[2,1,0]]
                    gt_img = gt_img[:,:,[2,1,0]]
            
                gt_img = gt_img[:h_old*args.scale, :w_old*args.scale, ...]

                avg_psnr_y += calculate_psnr(sr_img, gt_img, crop_border=args.scale, test_y_channel=True) * w
                avg_ssim_y += calculate_ssim(sr_img, gt_img, crop_border=args.scale, test_y_channel=True) * w
            # ------------------------------ #
            test_iter += 1
        
        if testset !='Personal':
            print("===> {} PSNR_Y: {:.4f}, SSIM_Y: {:.4f} AVG_TIME: {:.4f}".format(testset ,avg_psnr_y / len(test_data_loader), avg_ssim_y / len(test_data_loader), sum(end) / len(test_data_loader)))

if __name__ == "__main__":
    test(args.model_path, model_type='small')