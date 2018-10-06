import tensorflow as tf
import numpy as np
import cv2 as cv
import math,os,h5py,argparse,sys
from model import *

def main(args):
    dae_gan = DAE_GAN(args.batch, args.epoch, args.im_size, args.hw_size, args.k, args.alpa, args.beta, args.im_dir, args.save_dir, args.saveS_dir, args.saveT_dir)
    if args.is_train=='train':
        dae_gan.train()
    else:
        dae_gan.load(args.load_dir, args.raw_im_dir, args.save_im_dir)
    return True         

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_train', type=str, help='Training or loading.', default="train")   
    parser.add_argument('--load_dir', type=str, help='Load model checkpoint.', default="D:/SVHN_dataset/train/ckpt/")
    parser.add_argument('--raw_im_dir', type=str, help='Image to test.', default="D:/SVHN_dataset/test")
    parser.add_argument('--save_im_dir', type=str, help='Save sample images dir.', default="D:/SVHN_dataset/test_save/")

    parser.add_argument('--im_size', type=int, help='Image size (height, width) in pixels.', default=64)
    parser.add_argument('--batch', type=int, help='batch size.', default=64)
    parser.add_argument('--epoch', type=int, help='Number of training cyclese.', default=100)
    parser.add_argument('--hw_size', type=int, help='The size of the attention area removed.', default=30)
    parser.add_argument('--k', type=float, help='Gain coefficient of sigmoid when generating mask.', default=int(2e2))
    parser.add_argument('--alpa', type=float, help='Error weight_1.', default=0.4)
    parser.add_argument('--beta', type=float, help='Error weight_2.', default=0.6)
    parser.add_argument('--im_dir', type=str, help='Path to the image folder.', default="D:/SVHN_dataset/train/")
    parser.add_argument('--save_dir', type=str, help='Model save path.', default="D:/SVHN_dataset/train/ckpt/")
    parser.add_argument('--saveS_dir', type=str, help='The path that has been cleared.', default="D:/SVHN_dataset/train/SampleS/")
    parser.add_argument('--saveT_dir', type=str, help='Source image save path.', default="D:/SVHN_dataset/train/SampleT/")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
