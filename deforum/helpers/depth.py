import cv2
import math
import numpy as np
import os
import requests
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from einops import rearrange, repeat
from PIL import Image

from infer import InferenceHelper
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet

try:
    from numpngw import write_png
except ModuleNotFoundError:
    print(ModuleNotFoundError)
    import subprocess
    running = subprocess.run(['pip', 'install', 'numpngw'],stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(running)
    from numpngw import write_png

from tqdm import tqdm

def wget(url, outputdir):
    filename = url.split("/")[-1]

    ckpt_request = requests.get(url)
    request_status = ckpt_request.status_code

    # inform user of errors
    if request_status == 403:
        raise ConnectionRefusedError("You have not accepted the license for this model.")
    elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
    elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

    # write to model path
    with open(os.path.join(outputdir, filename), 'wb') as model_file:
        model_file.write(ckpt_request.content)


def download_file(url, models_path):
    filename = url.split("/")[-1]

    # Create the models_path directory if it does not exist
    os.makedirs(models_path, exist_ok=True)
    
    # Send a GET request to the URL
    response = requests.get(url, stream=True)
    
    # Get the total file size
    file_size = int(response.headers.get("Content-Length"))
    
    # Open a file in binary mode to write the content
    with open(os.path.join(models_path, filename), "wb") as f:
        # Initialize the progress bar
        pbar = tqdm(total=file_size, unit="B", unit_scale=True)
        
        # Iterate through the response data and write it to the file
        for data in response.iter_content(1024):
            f.write(data)
            # Update the progress bar manually
            pbar.update(len(data))
        
        # Close the progress bar
        pbar.close()


class DepthModel():
    def __init__(self, device):
        self.adabins_helper = None
        self.depth_min = 1000
        self.depth_max = -1000
        self.device = device
        self.midas_model = None
        self.midas_transform = None
    
    def load_adabins(self, models_path):
        if not os.path.exists(os.path.join(models_path,'AdaBins_nyu.pt')):
            print("..downloading AdaBins_nyu.pt")
            os.makedirs(models_path, exist_ok=True)
            download_file("https://huggingface.co/deforum/AdaBins/resolve/main/AdaBins_nyu.pt", models_path)
        self.adabins_helper = InferenceHelper(models_path, dataset='nyu', device=self.device)

    def load_midas(self, models_path, half_precision=True):
        if not os.path.exists(os.path.join(models_path, 'dpt_large-midas-2f21e586.pt')):
            print("..downloading dpt_large-midas-2f21e586.pt")
            download_file("https://huggingface.co/deforum/MiDaS/resolve/main/dpt_large-midas-2f21e586.pt", models_path)

        self.midas_model = DPTDepthModel(
            path=os.path.join(models_path, "dpt_large-midas-2f21e586.pt"),
            backbone="vitl16_384",
            non_negative=True,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.midas_transform = T.Compose([
            Resize(
                384, 384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()
        ])

        self.midas_model.eval()    
        if half_precision and self.device == torch.device("cuda"):
            self.midas_model = self.midas_model.to(memory_format=torch.channels_last)
            self.midas_model = self.midas_model.half()
        self.midas_model.to(self.device)

    def predict(self, prev_img_cv2, anim_args) -> torch.Tensor:
        w, h = prev_img_cv2.shape[1], prev_img_cv2.shape[0]

        # predict depth with AdaBins    
        use_adabins = anim_args.midas_weight < 1.0 and self.adabins_helper is not None
        if use_adabins:
            MAX_ADABINS_AREA = 500000
            MIN_ADABINS_AREA = 448*448

            # resize image if too large or too small
            img_pil = Image.fromarray(cv2.cvtColor(prev_img_cv2.astype(np.uint8), cv2.COLOR_RGB2BGR))
            image_pil_area = w*h
            resized = True
            if image_pil_area > MAX_ADABINS_AREA:
                scale = math.sqrt(MAX_ADABINS_AREA) / math.sqrt(image_pil_area)
                depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS) # LANCZOS is good for downsampling
                print(f"  resized to {depth_input.width}x{depth_input.height}")
            elif image_pil_area < MIN_ADABINS_AREA:
                scale = math.sqrt(MIN_ADABINS_AREA) / math.sqrt(image_pil_area)
                depth_input = img_pil.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
                print(f"  resized to {depth_input.width}x{depth_input.height}")
            else:
                depth_input = img_pil
                resized = False

            # predict depth and resize back to original dimensions
            try:
                with torch.no_grad():
                    _, adabins_depth = self.adabins_helper.predict_pil(depth_input)
                if resized:
                    adabins_depth = TF.resize(
                        torch.from_numpy(adabins_depth), 
                        torch.Size([h, w]),
                        interpolation=TF.InterpolationMode.BICUBIC
                    )
                    adabins_depth = adabins_depth.cpu().numpy()
                adabins_depth = adabins_depth.squeeze()
            except:
                print(f"  exception encountered, falling back to pure MiDaS")
                use_adabins = False
            torch.cuda.empty_cache()

        if self.midas_model is not None:
            # convert image from 0->255 uint8 to 0->1 float for feeding to MiDaS
            img_midas = prev_img_cv2.astype(np.float32) / 255.0
            img_midas_input = self.midas_transform({"image": img_midas})["image"]

            # MiDaS depth estimation implementation
            sample = torch.from_numpy(img_midas_input).float().to(self.device).unsqueeze(0)
            if self.device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)  
                sample = sample.half()
            with torch.no_grad():            
                midas_depth = self.midas_model.forward(sample)
            midas_depth = torch.nn.functional.interpolate(
                midas_depth.unsqueeze(1),
                size=img_midas.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            midas_depth = midas_depth.cpu().numpy()
            torch.cuda.empty_cache()

            # MiDaS makes the near values greater, and the far values lesser. Let's reverse that and try to align with AdaBins a bit better.
            midas_depth = np.subtract(50.0, midas_depth)
            midas_depth = midas_depth / 19.0

            # blend between MiDaS and AdaBins predictions
            if use_adabins:
                depth_map = midas_depth*anim_args.midas_weight + adabins_depth*(1.0-anim_args.midas_weight)
            else:
                depth_map = midas_depth

            depth_map = np.expand_dims(depth_map, axis=0)
            depth_tensor = torch.from_numpy(depth_map).squeeze().to(self.device)
        else:
            depth_tensor = torch.ones((h, w), device=self.device)
        
        return depth_tensor

    def save(self, filename: str, depth: torch.Tensor, bit_depth_output):
        depth = depth.cpu().numpy()
        if len(depth.shape) == 2:
            depth = np.expand_dims(depth, axis=0)
        self.depth_min = min(self.depth_min, depth.min())
        self.depth_max = max(self.depth_max, depth.max())
        print(f"  depth min:{depth.min()} max:{depth.max()}")
        denom = max(1e-8, self.depth_max - self.depth_min)
        denom_bitdepth_multiplier = {
            8: 255,
            16: 255 * 255,
            32: 1 # This one is 1 because 32bpc is float32 and isn't converted to uint, like 8bpc and 16bpc are
        }
        temp_image = rearrange((depth - self.depth_min) / denom * denom_bitdepth_multiplier[bit_depth_output], 'c h w -> h w c')
        temp_image = repeat(temp_image, 'h w 1 -> h w c', c=3)
        if bit_depth_output == 16:
            write_png(filename, temp_image.astype(np.uint16));
        elif bit_depth_output == 32:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
            cv2.imwrite(filename.replace(".png", ".exr"), temp_image)
        else: # 8 bit
            Image.fromarray(temp_image.astype(np.uint8)).save(filename)

