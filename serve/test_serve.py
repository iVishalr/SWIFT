from PIL import Image
import numpy as np
import requests
from torchvision import transforms

img_path = "/home/vishalr/Desktop/SWIFT/testsets/Set14/monarch.png"

transform = transforms.Compose([
                transforms.ToTensor(),
            ])

payload = {
    "data": open(img_path, 'rb').read(),
    "scale": 3,
}

res = requests.post("http://localhost:8080/predictions/swift", files=payload)
return_obj = res.json()

for images in return_obj:
    arr = np.array(return_obj[images], dtype=np.uint8)
    img = Image.fromarray(arr)
    img.show()