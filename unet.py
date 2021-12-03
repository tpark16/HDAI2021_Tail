import torch

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.transform import *
from utils.config import *
from net.model import *
from dataset.dataloader import * 

from tqdm import tqdm

def main():
    args = ParserArguments()

    trainer = nnUNetTrainer(args.pkl,args.model_weights)

    A2C_img,A4C_img,A2C_mask,A4C_mask = load_dataset(datapath=args.data_root,trainer=trainer)


    for n,(img,mask) in enumerate(tqdm(zip(A2C_img,A2C_mask))):

        mask_prob = trainer.network(img)

        # DICE & JACARD

        # export
        plt.imshow(img.detach().numpy()[0][0],cmap="gray")
        plt.imshow(mask_prob[0][0].detach().numpy(),alpha=.2)

        plt.savefig(f"exp/plots/A2C_{n}.png")


if __name__ == '__main__':
	main()
