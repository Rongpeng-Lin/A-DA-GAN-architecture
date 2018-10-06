import os,h5py
import numpy as np
import cv2 as cv

def clear_data(im_dir):
    name = im_dir+'digitStruct.mat'
    svhnMat = h5py.File(name=name, mode='r')
    im_names = [Im_name for Im_name in os.listdir(im_dir) if Im_name.split('.')[-1]=='png']  
    for im_name in im_names:
        im_num = int(im_name.split('.')[0])
        item = svhnMat['digitStruct']['bbox'][im_num-1].item()
        attr = svhnMat[item]['label']
        values = [svhnMat[attr.value[i].item()].value[0][0] for i in range(len(attr))] if len(attr) > 1 else [attr.value[0][0]]   
        for value in values:
            if value>=10.0:
                os.remove(im_dir+im_name)
                print('im is: ',im_name)
                print('value is: ',value)
                break
    return True

def create_data(raw_dir,if_clip):
    positive = raw_dir+'positive'
    negtive = raw_dir+'negtive'
    for new_dir in [positive,negtive]:
        os.makedirs(new_dir)   
    for im_name in os.listdir(raw_dir):
        if im_name.split('.')[-1]=='png':
            im = cv.imread(raw_dir+im_name)
            if if_clip:
                if_flip = np.random.uniform()
                if if_flip>0.5:
                    im_flip = cv.flip(im,1,dst=None)
                    cv.imwrite(negtive+'/'+im_name,im_flip)
                    os.remove(raw_dir+im_name)
                else:
                    cv.imwrite(positive+'/'+im_name,im)
                    os.remove(raw_dir+im_name)
            else:
                cv.imwrite(positive+'/'+im_name,im)
                im_flip = cv.flip(im,1,dst=None)
                cv.imwrite(negtive+'/'+im_name,im_flip)
                os.remove(raw_dir+im_name)
    return True
