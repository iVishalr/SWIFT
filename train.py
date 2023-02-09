import os
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utils
import skimage.color as sc
from util.calculate_psnr_ssim import calculate_psnr, calculate_ssim

from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, flop_count_table

from model.SWIFT import SWIFT
from data import Set5_val, dataset
from data.dataset import CPUPrefetcher, CUDAPrefetcher

parser = argparse.ArgumentParser(
    prog="train.py",
    description="Towards Faster and Efficient Lightweight Image Super Resolution using SwinV2 Transformers and Fourier Convolutions",
    formatter_class=argparse.MetavarTypeHelpFormatter,
)

# Training settings
parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for training. Default=2e-4")
parser.add_argument("--n_epochs", type=int, default=100000, help="Number of epochs to train model. By default model trains till 700K iterations.")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use for training. Default=64")
parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size to use for validation. Default=1")
parser.add_argument("--gamma", type=int, default=0.5, help="Learning rate decay factor. Default=0.5")
parser.add_argument("--step_size", type=int, default=100, help="learning rate decay per N epochs")

# Dataset Args
parser.add_argument("--root", type=str, default="./Datasets", help='Path to root of dataset directory. ')
parser.add_argument("--n_train", type=int, default=800, help="Number of samples in training set. Default=800")
parser.add_argument("--n_val", type=int, default=1, help="Number of images in validation set. Default=1")

# GPU Args
parser.add_argument("--cuda", action="store_true", default=True, help="Use Cuda enabled devices for training")
parser.add_argument("--threads", type=int, default=12, help="Number of workers for dataloader. Default=12")
parser.add_argument("--amp", action="store_true", default=False, help="Enables Automatic Mixed Precision for training.")
parser.add_argument("--load_mem", action="store_true", default=False, help="Loads entire dataset to RAM.")

# Model Checkpointing Args
parser.add_argument("--ckpt_dir", default="", type=str, help="Path to model checkpoint directory.")
parser.add_argument("--start_epoch", default=0, type=int, help="Epoch number to resume training.")
parser.add_argument("--log_every", type=int, default=100, help="Logs every 'x' iterations.")
parser.add_argument("--test_every", type=int, default=1000, help="Tests every 'x' iterations.")
parser.add_argument("--save_every", type=int, default=1000, help="Saves model every 'x' iterations.")
parser.add_argument("--pretrained", default="", type=str, help="Path to pretrained model.")
parser.add_argument("--resume", default="", type=str, help="Path to model checkpoint.")

# Model Args
parser.add_argument("--scale", type=int, default=2, help="Super Resolution scale. Scales: 2, 3, 4.")
parser.add_argument("--patch_size", type=int, default=128, help="Patch size to use for training. Patch Sizes: 128, 192, 256.")
parser.add_argument("--rgb_range", type=int, default=1, help="Maxium value of RGB.")
parser.add_argument("--n_colors", type=int, default=3, help="Number of color channels to use.")

# Miscelaneous Args
parser.add_argument("--seed", type=int, default=3407, help="Seed for reproducibility.")
parser.add_argument("--show_metrics", action="store_true", default=False, help="Enables PSNR and SSIM calculation for training set.")
parser.add_argument("--ext", type=str, default='.png', help="Image extension in the dataset.")
parser.add_argument("--model", type=str, default='SWIFT', help="Name for the model.")

args = parser.parse_args()

print("\nTraining Model: {}".format(args.model))
print("\nTraining Settings")
print("-" * 50)
print("Train Batch Size: {}".format(args.batch_size))
print("Test Batch Size: {}".format(args.test_batch_size))
print("Learning Rate: {}".format(args.lr))
print("Step Size: {}".format(args.step_size))
print("Gamma: {}".format(args.gamma))
print("Epochs: {}".format(args.n_epochs))
print("Start Epoch: {}".format(args.start_epoch))
print("Load From Memory: {}".format(args.load_mem))
print("Dataset Root: {}".format(args.root))
print("Seed Value: {}".format(args.seed))
print("Patch Size: {}".format(args.patch_size))
print("Pretrained Model: {}".format(args.pretrained))
print("Checkpoint Directory: {}".format(args.ckpt_dir))
print("Dataset Image Extension: {}".format(args.ext))
print("Scale: {}".format(args.scale))
print("Mixed Precision: {}".format(args.amp))
print("Device: {}".format(torch.cuda.get_device_name(0)))
print("-" * 50)
print()


writer = SummaryWriter(comment=f'{args.model}_x{args.scale}')

seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)

random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
if cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

print("===> Loading Datasets")

trainset = dataset.Dataset(args, args.load_mem)
testset = Set5_val.DatasetFromFolderVal("./testsets/Set5/HR/", "./testsets/Set5/LR/X{}/".format(args.scale), args.scale)

training_data_loader = DataLoader(dataset=trainset, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
testing_data_loader = DataLoader(dataset=testset, num_workers=args.threads, batch_size=args.test_batch_size, shuffle=False)

training_data_loader = CUDAPrefetcher(training_data_loader, device)
testing_data_loader = CUDAPrefetcher(testing_data_loader, device)

print("===> Building Model")
args.is_train = True

model = SWIFT(
    img_size=args.patch_size//args.scale,
    patch_size=1,
    in_channels=3,
    embd_dim=64,
    rfbs=[2, 2, 2, 2],
    depths=[2, 2, 2, 2],
    num_heads=[8, 8, 8, 8,8],
    mlp_ratio=1,
    window_size=8,
    residual_conv="3conv",
    scale=args.scale,
    act_layer=nn.GELU,
    feat_scale=False,
    attn_scale=True,
)

model = nn.DataParallel(model)

print("===> Setting Loss Function")
l1_criterion = nn.L1Loss()

print("===> Setting Optimizer")
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

print("===> Setting Scheduler")
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[384000, 534000, 584000, 609000], gamma=args.gamma)

print("===> Setting Checkpoint")

print("===> Setting GPU")
if cuda:
    model = model.to(device)
    l1_criterion = l1_criterion.to(device)

if args.amp:
    print("===> Setting Mixed Precision")
    scaler = torch.cuda.amp.GradScaler()

if args.pretrained:
    if os.path.isfile(args.pretrained):
        print("===> loading models '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if args.amp and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        args.start_epoch = checkpoint['epoch'] // len(training_data_loader) + 1
        print("===> loaded models '{}' (epoch {})".format(args.pretrained, checkpoint['epoch']))
    else:
        print("===> no models found at '{}'".format(args.pretrained))

def train_epochs(epoch):
    model.train()

    losses = []
    psnrs = []
    ssims = []
    psnrs_y = []
    ssims_y = []

    N = len(training_data_loader)
    training_data_loader.reset()
    
    data = training_data_loader.next()

    for it, _ in enumerate(range(len(training_data_loader)), 1):
        lr_tensor, hr_tensor = data["lr"], data["hr"]
        lr_tensor = lr_tensor.to(device, non_blocking=True)  # ranges from [0, 1]
        hr_tensor = hr_tensor.to(device, non_blocking=True)  # ranges from [0, 1]

        optimizer.zero_grad(set_to_none=True)

        if args.amp:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                sr_tensor = model(lr_tensor)
                content_loss = l1_criterion(sr_tensor, hr_tensor)
        else:
            sr_tensor = model(lr_tensor)
            content_loss = l1_criterion(sr_tensor, hr_tensor)

        loss = content_loss

        if args.show_metrics:
            predictions = utils.tensor2np(sr_tensor.detach()[0])
            ground_truth = utils.tensor2np(hr_tensor.detach()[0])

            psnr = utils.compute_psnr(predictions, ground_truth)
            ssim = utils.compute_ssim(predictions, ground_truth)

            psnr_y = utils.compute_psnr(utils.quantize(sc.rgb2ycbcr(predictions)[:, :, 0]), utils.quantize(sc.rgb2ycbcr(ground_truth)[:, :, 0]))
            ssim_y = utils.compute_ssim(utils.quantize(sc.rgb2ycbcr(predictions)[:, :, 0]), utils.quantize(sc.rgb2ycbcr(ground_truth)[:, :, 0]))

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        losses.append(loss.item())

        if (epoch * N + it) % args.log_every == 0:
            print(f"Epoch [{epoch}/{700000//N}] it [{epoch * N + it}/{N}] lr: {optimizer.param_groups[0]['lr']:e} loss: {loss.item():.5f}")

        # if args.show_metrics:
        #     psnrs.append(psnr)
        #     ssims.append(ssim)
        #     psnrs_y.append(psnr_y)
        #     ssims_y.append(ssim_y)
        #     # pbar.set_description(f"Epoch[{epoch}]({it}/{len(training_data_loader)}) lr: {optimizer.param_groups[0]['lr']:e} : Loss_l1: {loss.item():.5f} PSNR: {psnr:.5f} SSIM: {ssim:.5f} PSNR_Y: {psnr_y:.5f} SSIM_Y: {ssim_y:.5f}")
        # else:
        #     # pbar.set_description(f"Epoch[{epoch}] lr: {optimizer.param_groups[0]['lr']:e} : Loss_l1: {loss.item():.5f}")

        data = training_data_loader.next()

        if (epoch * N + it) % args.test_every == 0:
            with torch.no_grad():
                image_grid_HR = torchvision.utils.make_grid(
                    hr_tensor[:], normalize=True
                )
                image_grid_SR = torchvision.utils.make_grid(
                    sr_tensor[:], normalize=True
                )
                writer.add_image("HR Images", image_grid_HR, epoch * N + it)
                writer.add_image("SR Images", image_grid_SR, epoch * N + it)
                writer.add_scalar("L1 Loss", np.mean(losses), epoch * N + it)
            
            # perform validation
            valid(args.scale, epoch * N + it)
            model.train()

        if (epoch * N + it) % args.save_every == 0:
            save_checkpoint(epoch = epoch * N + it)

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
            forward_chop(model, patch, shave=shave, min_size=min_size) \
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

def valid(scale, epoch):
    model.eval()
    valid_iter = 1
    
    avg_psnr, avg_ssim = 0, 0
    avg_psnr_y, avg_ssim_y = 0, 0
    testing_data_loader.reset()
    batch = testing_data_loader.next()
    while batch is not None:
        lr_tensor, hr_tensor = batch["lr"], batch["hr"]
        _,_, h_old, w_old = lr_tensor.size()
        if args.cuda:
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

        with torch.no_grad():
            _, _, h_old, w_old = lr_tensor.size()
            h_pad = (h_old // model.module.window_size + 1) * model.module.window_size - h_old
            w_pad = (w_old // model.module.window_size + 1) * model.module.window_size - w_old
            lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
            pre = model(lr_tensor)
            pre = pre[..., :h_old * args.scale, :w_old * args.scale]

        with torch.no_grad():
            valid_image_grid_HR = torchvision.utils.make_grid(
                hr_tensor[:], normalize=True
            )
            valid_image_grid_SR = torchvision.utils.make_grid(
                pre[:], normalize=True
            )
        writer.add_image(f"Valid HR Image - {valid_iter}", valid_image_grid_HR, epoch)
        writer.add_image(f"Valid SR Image - {valid_iter}", valid_image_grid_SR, epoch)

        sr_img = utils.tensor2np(pre.detach()[0])
        gt_img = utils.tensor2np(hr_tensor.detach()[0])

        if sr_img.ndim == 3: 
            sr_img = sr_img[:,:,[2,1,0]]
            gt_img = gt_img[:,:,[2,1,0]]
        
        # crop GT 
        gt_img = gt_img[:h_old*args.scale, :w_old*args.scale, ...]

        avg_psnr += calculate_psnr(sr_img, gt_img, crop_border=args.scale)
        avg_ssim += calculate_ssim(sr_img, gt_img, crop_border=args.scale)

        avg_psnr_y += calculate_psnr(sr_img, gt_img, crop_border=args.scale, test_y_channel=True)
        avg_ssim_y += calculate_ssim(sr_img, gt_img, crop_border=args.scale, test_y_channel=True)

        valid_iter += 1
        batch = testing_data_loader.next()

    writer.add_scalar("Valid PSNR", avg_psnr/len(testing_data_loader), epoch)
    writer.add_scalar("Valid SSIM", avg_ssim/len(testing_data_loader), epoch)
    writer.add_scalar("Valid PSNR_Y", avg_psnr_y/len(testing_data_loader), epoch)
    writer.add_scalar("Valid SSIM_Y", avg_ssim_y/len(testing_data_loader), epoch)
    print("===> Valid. psnr: {:.4f}, ssim: {:.4f} psnr_y: {:.4f}, ssim_y: {:.4f}".format(avg_psnr / len(testing_data_loader), avg_ssim / len(testing_data_loader), avg_psnr_y / len(testing_data_loader), avg_ssim_y / len(testing_data_loader)))


def save_checkpoint(epoch):
    raw_model = model.module if hasattr(model, "module") else model
    model_folder = f"{args.ckpt_dir}/checkpoint_{args.model}_x{args.scale}/" 
    PATH = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    
    state_dict = {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }

    if args.amp:
        state_dict['scaler'] = scaler.state_dict()

    torch.save(state_dict, PATH)
    print("===> Checkpoint saved to {}".format(PATH))

def print_network(net: nn.Module):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    
    input_tensor = torch.randn(1,3,args.patch_size//args.scale,args.patch_size//args.scale, device=device)
    
    flops = FlopCountAnalysis(model, input_tensor)
    table = flop_count_table(flops)
    print("\nFLOP Analysis Table")
    print("-" * len(table.split("\n")[0]))
    print(table)
    print("-" * len(table.split("\n")[0]))
    print()

    print('\nTotal Number of FLOPs: {:.2f} G'.format(flops.total() / 1e9))
    print('\nTotal number of parameters: %d\n' % num_params)

print("===> Training")
print_network(model.module)
code_start = datetime.datetime.now()
timer = utils.Timer()

for epoch in range(args.start_epoch, (700000//len(training_data_loader))+1):
    t_epoch_start = timer.t()
    epoch_start = datetime.datetime.now()
    valid(args.scale, epoch)
    train_epochs(epoch)
    epoch_end = datetime.datetime.now()

    print('Epoch cost times: %s' % str(epoch_end-epoch_start))
    t = timer.t()
    prog = (epoch-args.start_epoch+1)/(args.n_epochs + 1 - args.start_epoch + 1)
    t_epoch = utils.time_text(t - t_epoch_start)
    t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
    print('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
    
code_end = datetime.datetime.now()
print('Code cost times: %s' % str(code_end-code_start))
