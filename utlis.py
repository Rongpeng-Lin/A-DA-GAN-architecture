import tensorflow as tf
import numpy as np
import cv2 as cv
import math,os,h5py,argparse,sys

def get_im_label(idx,batch,Dir,svhnMat):
    im_zeros = np.zeros([batch,64,64,3],np.float32)
    label_zeros = np.zeros([batch,10],np.float32)
    start = int(idx*batch)
    end = start+batch
    dirs = Dir[start:end]
    print('dirs: ',dirs)
    for i,Dir in enumerate(dirs):
        im_bgr = cv.resize(cv.imread(Dir),(64,64),interpolation=cv.INTER_CUBIC)
        im_rgb_unpro = im_bgr[:,:,::-1]
        im_rgb = ((im_rgb_unpro/255)-0.5)*2
        im_zeros[i,:,:,:] = im_rgb
        label_zeros[i,:] = im_dir2label(Dir,svhnMat)
    return im_zeros,label_zeros
        
def im_dir2label(a_dir,svhnMat):
    label = np.zeros([10,],np.float32)
    im_name = a_dir.split('/')[-1]
    im_num = int(im_name.split('.')[0])
        
    item = svhnMat['digitStruct']['bbox'][im_num-1].item()
    attr = svhnMat[item]['label']
    values = [svhnMat[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]   
    for value in values:
        label[int(value)] = 1.0
    return label

