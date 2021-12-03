from matplotlib import cm
import torch
import SimpleITK as sitk
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.transform import *
from utils.config import *
from net.model import *
from dataset.dataloader import * 
from utils.metric import *
from utils.file_op import *

from tqdm import tqdm

def main():
    args = ParserArguments()
    json_dict = dict()
    trainer = nnUNetTrainer(args.pkl,args.model_weights)

    A2C_img,A4C_img,A2C_mask,A4C_mask = load_dataset(datapath=args.data_root,trainer=trainer)


    for n,(img,mask) in enumerate(tqdm(zip(A2C_img,A2C_mask))):

        mask_prob = trainer.network(img)

        # sigmoid
        mask_prob_sig = torch.sigmoid(mask_prob)
        mask_prob_sig = mask_prob_sig[0][0].detach().numpy()
        mask_prob_sig = np.where(mask_prob_sig < 0.5, mask_prob_sig, 0)

        # negative
        mask_prob_np = mask_prob[0][0].detach().numpy()
        mask_prob_np = np.where(mask_prob_np < 0, mask_prob_np, 0)

        # print(mask_prob.shape, mask.shape)
        # print(mask_prob)
        # DICE & JACCARD

        dc_np = get_dice(mask, mask_prob_np)
        jc_np = get_jaccard(mask, mask_prob_np)

        dc_sig = get_dice(mask, mask_prob_sig)
        jc_sig = get_jaccard(mask, mask_prob_sig)


        id = 'A2C_' + str(n)
        json_dict[id] = {}

        # np: negative, sig = tensor sigmoid
        json_dict[id] = {'dice_np': dc_np, 'jaccard_np': jc_np, 'dice_sig': dc_sig, 'jaccard_sig': jc_sig}
        # write_json(json_dict, './exp/result.json')


        # export
        plt.imshow(img.detach().numpy()[0][0],cmap="gray")
        plt.imshow(mask_prob_np, alpha=.2)

        plt.savefig(f"exp/plots/A2C_{n}.png")
    
    print(json_dict)

    for n,(img,mask) in enumerate(tqdm(zip(A4C_img,A4C_mask))):

        mask_prob = trainer.network(img)
        mask_prob = mask_prob[0][0].detach().numpy()
        mask_prob = np.where(mask_prob < 0, mask_prob, 0)

        # DICE & JACCARD
        dc = get_dice(mask, mask_prob)
        jc = get_jaccard(mask, mask_prob)

        id = 'A4C_' + str(n)
        json_dict[id] = {}
        json_dict[id] = {'dice': dc, 'jaccard': jc}
        
        # export
        plt.imshow(img.detach().numpy()[0][0],cmap="gray")
        plt.imshow(mask_prob, alpha=.2)

        plt.savefig(f"exp/plots/A4C_{n}.png")

    print(json_dict)
    write_json(json_dict, './exp/result.json')

if __name__ == '__main__':
	main()
