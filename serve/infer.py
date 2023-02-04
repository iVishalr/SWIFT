import os
import argparse
import requests
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    prog="infer.py",
    description="Towards Faster and Efficient Lightweight Image Super Resolution using Swin Transformers and Fourier Convolutions",
    formatter_class=argparse.MetavarTypeHelpFormatter,
)
parser.add_argument("--path", type=str, required=True, help="Path to image for prediction.")
parser.add_argument("--scale", type=int, required=True, help="Super resolution scale. Scales: 2, 3, 4.")
parser.add_argument('--save', default=False, action="store_true", help='Store predictions.')
parser.add_argument("--save_dir", type=str, default="results", help="Path to folder for saving predicitons.")

args = parser.parse_args()

payload = {
    "data": open(args.path, 'rb').read(),
    "scale": args.scale,
}

if args.save:
    if args.save_dir == "results":
        if not os.path.exists(f"results/SWIFT_lightweight_x{args.scale}"):
            os.makedirs(os.path.join("results",f"SWIFT_lightweight_x{args.scale}"))
        save_path = os.path.join("results",f"SWIFT_lightweight_x{args.scale}")
    else:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir

res = requests.post("http://localhost:8080/predictions/swift", files=payload)
return_obj = res.json()
img_name = os.path.basename(args.path)[:-4]

for images in return_obj:
    arr = np.array(return_obj[images], dtype=np.uint8)
    img = Image.fromarray(arr)
    
    if not args.save:
        img.show()
    else:
        img.save(os.path.join(save_path, f"{img_name}_SWIFT.png"))