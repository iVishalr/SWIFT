import os
import numpy as np
import torch
import zipfile
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from torchvision.utils import save_image

class Handler(BaseHandler):
    def __init__(self):
        self.initialized = False
        self.map_location = None
        self.device = None
        self.use_gpu = True
        self.swift_models: dict = {}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])

    def initialize(self, context):
        """
        Extract the models zip and load the serialized model
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")

        self.map_location, self.device, self.use_gpu = \
            ("cuda", torch.device("cuda:"+str(gpu_id)), True) if torch.cuda.is_available() else \
            ("cpu", torch.device("cpu"), False)

        if not os.path.exists(os.path.join(model_dir, "model")):
            with zipfile.ZipFile(os.path.join(model_dir, "models.zip"), "r") as zip_handler:
                zip_handler.extractall(model_dir)

        # build model object
        from model.SWIFT import swift_x2 as SWIFTx2, swift_x3 as SWIFTx3, swift_x4 as SWIFTx4
        self.swift_models["x2"] = SWIFTx2()
        self.swift_models["x3"] = SWIFTx3()
        self.swift_models["x4"] = SWIFTx4()
        
        param_key_g = "model"
        
        for models in self.swift_models:
            state_dict = torch.load(os.path.join(model_dir, "model_zoo", "SWIFT", f"SWIFT-S-{models[-1]}x.pth"), map_location=self.map_location)
            self.swift_models[models].load_state_dict(state_dict[param_key_g] if param_key_g in state_dict.keys() else state_dict, strict=True)
            self.swift_models[models].eval()
            self.swift_models[models].to(self.device)

        self.initialized = True

    def _preprocess(self, req):

        data = req.get("data")
        if data is None:
            data = req.get("body")

        scale = int(req.get("scale"))

        data = Image.open(io.BytesIO(data))
        data = self.transform(data)

        if scale > 4:
            scale = 4

        win_size = self.swift_models[f"x{scale}"].window_size
        
        if len(data.size()) == 3:
            data = data.unsqueeze(0) # add batch dimension
        
        _, _, h_old, w_old = data.size()
        input_dims = (h_old, w_old) # keep track of the input dims
        h_pad = (h_old // win_size + 1) * win_size - h_old
        w_pad = (w_old // win_size + 1) * win_size - w_old
        data = torch.cat([data, torch.flip(data, [2])], 2)[:, :, :h_old + h_pad, :]
        data = torch.cat([data, torch.flip(data, [3])], 3)[:, :, :, :w_old + w_pad]
        input_data = {"data": data, "scale": scale, "input_dims": input_dims}

        return input_data

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self._preprocess(req) for req in requests]
        tensors = []
        metadata = []

        for img in images:
            tensors.append(img["data"])
            metadata.append((img["scale"], img["input_dims"]))

        res = {}
        res["tensors"] = tensors
        res["metadata"] = metadata
        return res

    def inference(self, data, *args, **kwargs) -> torch.Tensor:
        tensors = data["tensors"]
        metadata = data["metadata"]
        predictions = []
        
        for meta, tensor in zip(metadata,tensors):
            scale = int(meta[0])
            h_old, w_old = meta[1]
            model = self.swift_models[f"x{scale}"]

            with torch.no_grad():
                marshalled_data = tensor.to(self.device)
                prediction = self._forward_chop(model, marshalled_data, scale)
                h_old, w_old = int(h_old), int(w_old)
                op = prediction[:, :, :h_old*scale, :w_old*scale]
                op = op.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                op = (op * 255.0).round().astype(np.uint8)
                op = np.transpose(op, (1, 2, 0))
                predictions.append(op)

        return predictions

    def postprocess(self, output_batch):
        """
        Create an image(jpeg) using the output tensor.
        """
        postprocessed_data = {}
        for ix,op in enumerate(output_batch):
            postprocessed_data[f"image{ix}"] = op.tolist()

        return [postprocessed_data]

    def _forward_chop(self, model, x: torch.Tensor, scale: int, shave: int = 10, min_size: int = 50000):
        # Using forward_chop inference to make it suitable for inference on very large images (>512), otherwise it leads to OOM
        n_GPUs = 1
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
                self._forward_chop(model, patch, scale, shave=shave, min_size=min_size) \
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