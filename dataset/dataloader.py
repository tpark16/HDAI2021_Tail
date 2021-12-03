from glob import glob
import os
import numpy as np
import torch

from utils.transform import *

def load_dataset(datapath,trainer):
    A2C_img = []
    A4C_img = []
    A2C_mask = []
    A4C_mask = []

    for n,img_path in enumerate(glob(os.path.join(datapath,"A2C","*.png"))):
        img = cv2.imread(img_path, 0)
        img = center_crop(img, trainer.patch_size[0],trainer.patch_size[1])
        img = image_minmax(img)
        img = cv2.resize(img, ( trainer.patch_size[1],trainer.patch_size[0]))
        # Add channel axis
        img = img[None,None, ...].astype(np.float32)
        
        img_torch = torch.from_numpy(img)
        A2C_img.append(img_torch)

        mask = np.load(img_path.replace(".png",".npy"))
        mask = center_crop(mask, trainer.patch_size[0],trainer.patch_size[1])
        mask = cv2.resize(mask, ( trainer.patch_size[1],trainer.patch_size[0]))

        A2C_mask.append(mask)


    for n,img_path in enumerate(glob(os.path.join(datapath,"A4C","*.png"))):
        img = cv2.imread(img_path, 0)
        img = center_crop(img, trainer.patch_size[0],trainer.patch_size[1])
        img = image_minmax(img)
        img = cv2.resize(img, ( trainer.patch_size[1],trainer.patch_size[0]))
        # Add channel axis
        img = img[None,None, ...].astype(np.float32)
        
        img_torch = torch.from_numpy(img)
        A4C_img.append(img_torch)

        mask = np.load(img_path.replace(".png",".npy"))
        mask = center_crop(mask, trainer.patch_size[0],trainer.patch_size[1])
        mask = cv2.resize(mask, ( trainer.patch_size[1],trainer.patch_size[0]))

        A4C_mask.append(mask)

    print(f"load {len(A2C_img)} A2C images and {len(A4C_img)} A4C images!")

    return A2C_img,A4C_img,A2C_mask,A4C_mask
