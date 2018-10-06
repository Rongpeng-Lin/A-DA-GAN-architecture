import tensorflow as tf
import numpy as np
import cv2 as cv
import math,os,h5py,argparse,sys
from Create_data import clear_and_create

def main(args):
    if args.op_type=='clear':
        clear_and_create.clear_data(args.im_dir)
        return True
    else:
        clear_and_create.create_data(args.raw_dir,args.if_clip)
        return True         

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--op_type', type=str, help='Choose to clear data or create positive and negative samples,clear or create.', default="clear")
    parser.add_argument('--im_dir', type=str, help='Path to the image folder.', default="D:/SVHN_dataset/train/")
    parser.add_argument('--raw_dir', type=str, help='The path that has been cleared.', default="D:/SVHN_dataset/train/")
    parser.add_argument('--if_clip', type=bool,help='Whether to divide the picture into two parts (the training set will be reduced after segmentation).', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
