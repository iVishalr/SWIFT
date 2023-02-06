import os
import copy
import utils
import torch
import random
import argparse

import torch.nn as nn
from torch.utils.data import DataLoader

from model.SWIFT import SWIFT
from data.testset import TestSet

from PIL import Image
from util.calculate_psnr_ssim import calculate_psnr, calculate_ssim
from fvcore.nn import FlopCountAnalysis, flop_count_table

parser = argparse.ArgumentParser(
    prog="test.py",
    description="Towards Faster and Efficient Lightweight Image Super Resolution using SwinV2 Transformers and Fourier Convolutions",
    formatter_class=argparse.MetavarTypeHelpFormatter,
)
parser.add_argument("--scale", type=int, required=True, help="Super resolution scale. Scales: 2, 3, 4.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the trained SWIFT model.")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size to use for testing. Default=1.")
parser.add_argument("--cuda", action="store_true", default=False, help="Use CUDA enabled device to perform testing.")
parser.add_argument('--jit', default=False, action="store_true", help='Perform inference using JIT.')
parser.add_argument("--forward_chop", action="store_true", default=False, help="Use forward_chop for performing inference on devices with less memory.")
parser.add_argument("--seed", type=int, default=3407, help="Seed for reproducibility.")
parser.add_argument("--summary", action="store_true", default=False,help="Print summary table for model.")

args = parser.parse_args()

seed = args.seed
if seed is None:
    seed = random.randint(1, 10000)

random.seed(seed)
torch.manual_seed(seed)

cuda = args.cuda
device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

device_str = None
if cuda and torch.cuda.is_available():
    device_str = torch.cuda.get_device_name(0) + " GPU"
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
else:
    device_str = "CPU"

print(f"-> Running Testing on {device_str}.")

dataset_path = "./testsets/"

testset_Set5 = TestSet(f"{dataset_path}Set5/HR/", None, args.scale)
testset_Set14 = TestSet(f"{dataset_path}Set14/", None, args.scale)
testset_BSDS100 = TestSet(f"{dataset_path}BSD100/", None, args.scale)
testset_Urban100 = TestSet(f"{dataset_path}Urban100/", None, args.scale)
testset_Manga109 = TestSet(f"{dataset_path}Manga109/", None, args.scale)
testset_General100 = TestSet(f"{dataset_path}General100/", None, args.scale)

Set5_data_loader = DataLoader(dataset=testset_Set5, num_workers=0, batch_size=args.batch_size, shuffle=False)
Set14_data_loader = DataLoader(dataset=testset_Set14, num_workers=0, batch_size=args.batch_size, shuffle=False)
BSDS100_data_loader = DataLoader(dataset=testset_BSDS100, num_workers=0, batch_size=args.batch_size, shuffle=False)
Urban100_data_loader = DataLoader(dataset=testset_Urban100, num_workers=0, batch_size=args.batch_size, shuffle=False)
Manga109_data_loader = DataLoader(dataset=testset_Manga109, num_workers=0, batch_size=args.batch_size, shuffle=False)
General100_data_loader = DataLoader(dataset=testset_General100, num_workers=0, batch_size=args.batch_size, shuffle=False)

test_data_loader_dict = {
    "Set5": Set5_data_loader, 
    "Set14": Set14_data_loader, 
    "BSD100": BSDS100_data_loader, 
    "Urban100": Urban100_data_loader, 
    "Manga109": Manga109_data_loader, 
    "General100": General100_data_loader
}

def test(model_path):

    base_model = SWIFT(
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
    
    model_paths = [
        (model_path, 1), #for ensemble prediction, add more here (model_path, weight)
    ]

    models = []
    i = 1
    for model_path, model_weights in model_paths:
        model = copy.deepcopy(base_model)
        print(f"-> Loading Model-{i} from", model_path)
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'], strict=True)

        model.to(device)
        model.eval()

        if args.summary:
            print(f"-> Printing Model-{i} Summary.")
            print_network(model)

        if args.jit:
            print(f"-> Using JIT for Optimizing Model-{i} Inference.")

            x = torch.randn(4,3,64,76, dtype=torch.float32, device=device)
            y = torch.randn(4,64,64,76, dtype=torch.float32, device=device)
            inp1 = torch.randn(4,32,256,181, dtype=torch.float32, device=device)
            x_h = torch.randn(4,32,256,181, dtype=torch.float32, device=device)
            x_l = torch.randn(4,32,256,181, dtype=torch.float32, device=device)

            model.head = torch.jit.script(model.head, example_inputs=[(x)])
            model.conv_after_body = torch.jit.script(model.conv_after_body, example_inputs=[(y)])
            model.conv_before_upsample = torch.jit.script(model.conv_before_upsample, example_inputs=[(y)])
            model.tail = torch.jit.script(model.tail, example_inputs=[(y)])

            for i, layers in enumerate(model.layers):
                # select RFB from each layer and optimise non ffc parts
                for j, rfb in enumerate(layers.rfbs):
                    model.layers[i].rfbs[j].downsample = torch.jit.script(rfb.downsample, example_inputs=[(inp1)])
                    model.layers[i].rfbs[j].extractor_body = torch.jit.script(rfb.extractor_body, example_inputs=[(inp1)])
                    model.layers[i].rfbs[j].conv1x1 = torch.jit.script(rfb.conv1x1, example_inputs=[(inp1)])
                    model.layers[i].rfbs[j].scam = torch.jit.script(rfb.scam, example_inputs=[x_h, x_l])
            print("-> JIT Optimization Completed.")

        models.append((model, model_weights))
        i+=1

    print(f"-> Model(s) Built for Testing on {device_str}.")

    timer = utils.Timer()
    timer.to("cuda" if cuda and torch.cuda.is_available() else "cpu")

    print("-> Testing Started.")
    
    testset_results = []

    for testset, test_data_loader in test_data_loader_dict.items():

        elapsed_time = []
        avg_psnr_y, avg_ssim_y = 0, 0
        print(f"\n-> Testing SWIFT(x{args.scale}) on {testset} dataset.")

        for i, batch in enumerate(test_data_loader):
            lr_tensor, hr_tensor, path = batch["lr"], batch["hr"], batch["hr_path"]
            path = os.path.basename(path[0])
            _,_, h_old, w_old = lr_tensor.size()
            
            lr_tensor = lr_tensor.to(device)
            hr_tensor = hr_tensor.to(device)

            with torch.no_grad():
                _, _, h_old, w_old = lr_tensor.size()
                h_pad = (h_old // model.window_size + 1) * model.window_size - h_old
                w_pad = (w_old // model.window_size + 1) * model.window_size - w_old
                lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
                lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

                if args.forward_chop:
                    # saves memory on during testing on very large images
                    timer.record()
                    for model,w in models:
                        pre = forward_chop(model, lr_tensor, args.scale) 
                        pre = pre[..., :h_old * args.scale, :w_old * args.scale]
                        pred.append((pre,w))
                    timer.stop()
                else:
                    timer.record()
                    pre = None
                    pred = []
                    for model, w in models:
                        pre = model(lr_tensor)
                        pre = pre[..., :h_old * args.scale, :w_old * args.scale]
                        pred.append((pre,w))
                    timer.stop()
                
                timer.sync()
                elapsed_time.append(timer.get_elapsed_time())

            for pre,w in pred:
                sr_img = utils.tensor2np(pre.detach()[0])
                gt_img = utils.tensor2np(hr_tensor.detach()[0])

                if sr_img.ndim == 3: 
                    sr_img = sr_img[:,:,[2,1,0]]
                    gt_img = gt_img[:,:,[2,1,0]]
            
                gt_img = gt_img[:h_old*args.scale, :w_old*args.scale, ...]

                psnr = calculate_psnr(sr_img, gt_img, crop_border=args.scale, test_y_channel=True) 
                ssim = calculate_ssim(sr_img, gt_img, crop_border=args.scale, test_y_channel=True)
                avg_psnr_y += psnr * w
                avg_ssim_y += ssim * w

            print(f"Testing {i+1} {path}\t- PSNR_Y: {psnr:.2f} dB; SSIM_Y: {ssim:.4f}; Inference Time: {elapsed_time[-1]:.2f}ms")
        
        elapsed_time = elapsed_time[1:] # To remove the initial warmup inference times
        avg_psnry = avg_psnr_y / len(test_data_loader)
        avg_ssimy = avg_ssim_y / len(test_data_loader)
        avg_time = sum(elapsed_time) / len(elapsed_time)

        heading = f"\nAverage Results for {testset}"
        avg_results = [testset, avg_psnry, avg_ssimy, avg_time]
        testset_results.append(avg_results)

        print(heading)
        print("-"*len(heading))
        print("PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}; Testing Time: {:.2f}ms".format(avg_psnry, avg_ssimy, avg_time))

    # fancy print
    description = "\nSWIFT Benchmark Results"
    dline = "=" * 82
    line = "-" * 82
    print(description+"\n"+dline)
    print(f"\nScale: x{args.scale} · Number of Parameters: {sum([p.numel() for p in model.parameters()])} · Device: {device_str}.\n")
    print(line)
    for res in testset_results:
        print("{: <10} - PSNR_Y: {:.4f} dB; SSIM_Y: {:.4f}; Inference Time: {:.2f}ms".format(*res))
    print(dline)
    print("\n-> Testing Completed.")

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

if __name__ == "__main__":
    test(args.model_path)