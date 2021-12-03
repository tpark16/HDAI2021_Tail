import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def ParserArguments():
    args = argparse.ArgumentParser()

    # Directory Setting 
    args.add_argument('--data_root', type=str, default='./data/', help='dataset directory')
    args.add_argument('--pkl', type=str, default="exp/model_final_checkpoint.model.pkl", help='model pkl file')
    args.add_argument('--model_weights', type=str, default="exp/model_final_checkpoint.model", help='model weight file')

    args = args.parse_args()

    print("\n")
    print("------------Parameters--------------")
    print(f"- Data root: {args.data_root}")
    print(f"- Model pkl file: {args.pkl}")
    print(f"- Model weights file: {args.model_weights}")
    print("------------Parameters--------------")
    print("\n")

    return args
