import os

import torch
import yaml
import folder_paths
import comfy.model_management as mm
import comfy.utils
import numpy as np
import cv2
from tqdm import tqdm

from PIL import Image

from PIL.PngImagePlugin import PngInfo


import os
import os.path as osp
import tyro
import subprocess
from .src.config.argument_config import ArgumentConfig
from .src.config.inference_config import InferenceConfig
from .src.config.crop_config import CropConfig#xpose
from .src.live_portrait_pipeline import LivePortraitPipeline



class InferenceConfig:
    def __init__(
        self,
        
        flag_use_half_precision=True,
        flag_lip_zero=True,
        lip_zero_threshold=0.03,
        flag_eye_retargeting=False,
        flag_lip_retargeting=False,
        flag_stitching=True,
        input_shape=(256, 256),
        device_id=0,
        flag_do_rot=True,
        **kwargs,
    ):
        self.flag_use_half_precision = flag_use_half_precision
        self.flag_lip_zero = flag_lip_zero
        self.lip_zero_threshold = lip_zero_threshold
        self.flag_eye_retargeting = flag_eye_retargeting
        self.flag_lip_retargeting = flag_lip_retargeting
        self.flag_stitching = flag_stitching
        self.input_shape = input_shape
        self.device_id = device_id
        self.flag_do_rot = flag_do_rot
        
        
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})
 
class LivePortraitp2p:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {

            "source_image": ("IMAGE",),
            "driving_image": ("IMAGE",),
            
            }
        }

    RETURN_TYPES = (
        "IMAGE",
    )
    RETURN_NAMES = (
        "images output",
    )
    FUNCTION = "process"
    CATEGORY = "LivePortrait"

    def process(
        self,
        source_image,
        driving_image,
        **kwargs
    ):
        print("----xu0---")
        for (batch_number, image) in enumerate(source_image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
        
            metadata = PngInfo()           
            img.save("custom_nodes/Comfyui-Liveportrait_v3/assets/examples/source/s12.jpg", pnginfo=metadata, compress_level=4)

        for (batch_number, image) in enumerate(driving_image):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
        
            metadata = PngInfo()           
            img.save("custom_nodes/Comfyui-Liveportrait_v3/assets/examples/source/d9.jpg", pnginfo=metadata, compress_level=4)


        tyro.extras.set_accent_color("bright_cyan")
        print("----xu00---")
        args = ArgumentConfig()#tyro.cli(ArgumentConfig)

        print(args)
        print("----xu---")
        #raise ValueError(1)
        
        
        
        inference_cfg = partial_fields(InferenceConfig, args.__dict__)
        crop_cfg = partial_fields(CropConfig, args.__dict__)
        
        live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )
    # run
        
        result = live_portrait_pipeline.execute(args)
        
        
        #raise ValueError(result)
        # result = np.array(result) 
        # result = torch.from_numpy(result).float() / 255.0
        
        
        return (result,)
    

        

NODE_CLASS_MAPPINGS = { 
    "LivePortraitp2p": LivePortraitp2p,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LivePortraitp2p": "LivePortraitp2p",
    }
