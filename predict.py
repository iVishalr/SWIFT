import os
import cv2
import glob
import argparse
import requests
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from util import calculate_psnr_ssim as util
from model.effcNet import EFFCNet

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def forward_chop(model, x, scale, shave=10, min_size=30000):
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=2, help='scale factor: 2, 3, 4')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training EFFCNet. '
                                       'Just used to differentiate two different settings in Table 2 of the paper. '
                                       'Images are NOT tested patch by patch.')
    parser.add_argument('--model_path', type=str,default='model_zoo/effcnet/EFFCNet-S-2x.pth')
    parser.add_argument('--folder_lq', type=str, default=None, help='input low-quality test image folder')
    parser.add_argument('--folder_gt', type=str, default=None, help='input ground-truth test image folder')
    parser.add_argument('--tile', type=int, default=None, help='Tile size, None for no tile during testing (testing as a whole)')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
    parser.add_argument('--jit', default=False, action="store_true", help='Perform inference using JIT.')
    parser.add_argument('--forward_chop', default=False, action="store_true", help='Use Forward Chop.')


    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    # set up model
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    model = define_model(args)
    model.eval()
    model = model.to(device)

    if args.jit:
        print("Using JIT")
        model = torch.jit.script(model, example_inputs=(torch.randn(1,3,256,256, dtype=torch.float32, device=device),))

    folder, save_dir, border, window_size = setup(args)
    os.makedirs(save_dir, exist_ok=True)
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    psnr, ssim, psnr_y, ssim_y, avg_time = 0, 0, 0, 0, []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    gpu_memory = []
    
    img_gt, img_lq = None, None

    for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):

        imgname, img_lq, img_gt = get_image_pair(args, path, True if ('DIV2K' in path) or ('Set5' in path) else False)  # image to HWC-BGR, float32

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
            
            if args.forward_chop:
                start.record()
                output = forward_chop(model, img_lq, scale=args.scale)
                end.record()
                gpu_memory.append(torch.cuda.max_memory_allocated() // 1024 // 1024)
            else:
                start.record()
                output = test(img_lq, model, args, window_size)
                end.record()
                gpu_memory.append(torch.cuda.max_memory_allocated() // 1024 // 1024)

            torch.cuda.current_stream().synchronize()
            avg_time.append(start.elapsed_time(end))
            output = output[..., :h_old * args.scale, :w_old * args.scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        cv2.imwrite(f'{save_dir}/{imgname}_EFFC.png', output)

        # evaluate psnr/ssim
        if img_gt is not None:
            img_gt = (img_gt * 255.0).round().astype(np.uint8)  # float32 to uint8
            img_gt = img_gt[:h_old * args.scale, :w_old * args.scale, ...]  # crop gt
            img_gt = np.squeeze(img_gt)

            psnr = util.calculate_psnr(output, img_gt, crop_border=border)
            ssim = util.calculate_ssim(output, img_gt, crop_border=border)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            if img_gt.ndim == 3:  # RGB image
                psnr_y = util.calculate_psnr(output, img_gt, crop_border=border, test_y_channel=True)
                ssim_y = util.calculate_ssim(output, img_gt, crop_border=border, test_y_channel=True)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)

            print('Testing {:d} {:20s} - PSNR: {:.2f} dB; SSIM: {:.4f};'
                  ' PSNR_Y: {:.2f} dB; SSIM_Y: {:.4f}; Inference Time: {:.2f}ms'.
                  format(idx+1, imgname, psnr, ssim, psnr_y, ssim_y, (start.elapsed_time(end))))
        else:
            print('Testing {:d} {:20s} Inference time: {:.2f}ms'.format(idx+1, imgname, (start.elapsed_time(end))))
    
    # summarize psnr/ssim
    if img_gt is not None:
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        avg_time = avg_time[2:]
        print('\n{} \n-- Average PSNR/SSIM(RGB): {:.2f} dB; {:.4f} AVG Time: {:.2f}ms'.format(save_dir, ave_psnr, ave_ssim, sum(avg_time) / len(avg_time)))
        print("Max GPU Memory:", max(gpu_memory))
        if img_gt.ndim == 3:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            print('-- Average PSNR_Y/SSIM_Y: {:.2f} dB; {:.4f}'.format(ave_psnr_y, ave_ssim_y))


def define_model(args):

    effcnet_model = EFFCNet(
        img_size=64,
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

    param_key_g = "model"
    model = effcnet_model
    pretrained_model = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    return model


def setup(args):
    # 001 classical image sr/ 002 lightweight image sr
    save_dir = f'results/effc_lightweight_x{args.scale}'
    folder = args.folder_gt if args.folder_gt else args.folder_lq
    
    if not os.path.exists(folder):
        print(f"Folder: {folder} does not exists.")
        exit(1)

    border = args.scale
    window_size = 8

    return folder, save_dir, border, window_size


def get_image_pair(args, path, attach_scale=True):
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    if args.folder_gt is not None and os.path.exists(args.folder_gt):
        img_gt = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    else:
        img_gt = None

    if args.folder_lq is not None and os.path.exists(args.folder_lq):
        if attach_scale:
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}x{args.scale}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
        else:
            img_lq = cv2.imread(f'{args.folder_lq}/{imgname}{imgext}', cv2.IMREAD_COLOR).astype(np.float32) / 255.
    else:
        img_lq = img_gt
        img_gt = None

    if img_lq is None and img_gt is None:
        print("Both HR and LR images cannot be None")
        exit(1)

    return imgname, img_lq, img_gt


def test(img_lq, model, args, window_size):
    if args.tile is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(args.tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args.tile_overlap
        sf = args.scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        output = E.div_(W)

    return output

if __name__ == '__main__':
    main()
