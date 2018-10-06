import tensorflow as tf
import numpy as np
import cv2 as cv
import math,os,h5py,argparse,sys

def conv(name,x,kers,s,outs,pad):
    with tf.variable_scope(name):
        ker = int(math.sqrt(kers))
        shape = [i.value for i in x.get_shape()]
        w = tf.get_variable('w',
                            [ker,ker,shape[-1],outs],
                            tf.float32,
                            tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b',[outs],tf.float32,tf.constant_initializer(0.))
        padd = "SAME" if pad else "VALID"
        x_conv = tf.nn.conv2d(x,w,[1,s,s,1],padd) + b
        return x_conv

def res_block(name,x):
    with tf.variable_scope(name):
        shape = [i.value for i in x.get_shape()]
        conv1 = conv(name+'_conv1',x,3*3,shape[-1],1,True)
        bn1 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1,scale=True,updates_collections=None))    
        conv2 = conv(name+'_conv2',bn1,3*3,shape[-1],1,True)
        bn2 = tf.contrib.layers.batch_norm(conv2,scale=True,updates_collections=None)
        return tf.nn.relu(bn2+x)       
    
def conv_trans(name,x,sizes,s,outs,ifpad):
    with tf.variable_scope(name):
        ker = int(math.sqrt(sizes))
        shape = [i.value for i in x.get_shape()]
        ins = shape[-1]//4
        w = tf.get_variable('w',[ker,ker,ins,outs],tf.float32,tf.contrib.layers.xavier_initializer_conv2d())
        b = tf.get_variable('b',[outs],tf.float32,tf.constant_initializer(0.))
        pad = "SAME" if ifpad else "VALID"
        x_conv = tf.nn.conv2d(tf.depth_to_space(x,2),w,[1,s,s,1],pad)+b
        return x_conv
    
def lrelu(name,x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)

def tanh(name,x):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)
        
def BN(name,x):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(x,scale=True,updates_collections=None)
    
def FC_location(name,x,outs,im_size,hw_size):
    with tf.variable_scope(name):
        raw_shape = [i.value for i in x.get_shape()]
        new_shape = int(raw_shape[1]*raw_shape[2]*raw_shape[3])
        x_resh = tf.reshape(x,[-1,new_shape])
        w = tf.get_variable('w',[new_shape,outs],tf.float32,tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',[1,outs],tf.float32,tf.constant_initializer(0.))
        fc_ = tf.matmul(x_resh,w)+b
        # Add a range limit to the position coordinates while making the gradient softer.
        xy_constraint = tf.nn.sigmoid(fc_)*(im_size-hw_size)
        return tf.cast(tf.round(xy_constraint),tf.int32)

def get_constant(im_size):
    zero_x = np.zeros([im_size,im_size],np.float32)
    zero_y = np.zeros([im_size,im_size],np.float32)
    for i in range(im_size):
        zero_x[:,i] = i+1 
        zero_y[i,:] = i+1
    return zero_x,zero_y

def sigmoid_mask(x,k):
    k = int(k)
    X = tf.cast(x,tf.float32)
    return 1/(1+tf.exp(-1*k*X))
    
def get_mask(xy,reigon_w,reigon_h,im_size,k,B): 
    # xy: [batch, 2]: coordinates
    # reigon_w, reigon_h: size of the area
    # im_size: Image size for generating the original X
    # k: The growth factor of the sigmoid function
    with tf.variable_scope('Mask'):
        x_left = tf.expand_dims(tf.expand_dims(tf.expand_dims(xy[:,0],1),2),3)
        x_right = tf.expand_dims(tf.expand_dims(tf.expand_dims(xy[:,0]+reigon_w,1),2),3)
        y_top = tf.expand_dims(tf.expand_dims(tf.expand_dims(xy[:,1],1),2),3)
        y_bottom = tf.expand_dims(tf.expand_dims(tf.expand_dims(xy[:,1]+reigon_h,1),2),3)
        x_value,y_value = get_constant(im_size)
        x_constant = np.tile(np.expand_dims(np.expand_dims(x_value,0),3),[B,1,1,1])
        y_constant = np.tile(np.expand_dims(np.expand_dims(y_value,0),3),[B,1,1,1])
        A = sigmoid_mask(x_constant-x_left,k)
        C = sigmoid_mask(x_constant-x_right,k)
        D = sigmoid_mask(y_constant-y_top,k)
        E = sigmoid_mask(y_constant-y_bottom,k)
        return (A-C)*(D-E)

def FC(name,x,outs):
    with tf.variable_scope(name):
        shape = [i.value for i in x.get_shape()]
        size = int(shape[1]*shape[2]*shape[3])
        x_reshape = tf.reshape(x,[-1,size])
        w = tf.get_variable('w',[size,outs],tf.float32,tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',[1,outs],tf.float32,tf.constant_initializer(0.))
        return tf.nn.sigmoid(tf.matmul(x_reshape,w)+b)
       
def resnet_classifer(name,x):  #  x: batch，4,4,512
    with tf.variable_scope(name):
        res1 = res_block('res1',x)
        res2 = res_block('res2',res1)
        res3 = res_block('res3',res2)
        res4 = res_block('res4',res3)
        res5 = res_block('res5',res4)
        res6 = res_block('res6',res5)
        res7 = res_block('res7',res6)
        probs = FC('probs',res7,10)
        return probs

def One_ep_Iter(im_dir,batch):
    num_ims = min(len(os.listdir(im_dir+'positive')),len(os.listdir(im_dir+'negtive')))
    return num_ims//batch

def get_shape(x):
    L = [i.value for i in x.get_shape()]
    return L

def save_im(sample_im,save_dir,cur_ep,cur_batch,batch):
    for i in range(batch):
        im_S = sample_im[i,:,:,:]
        im_s = (im_S[:,:,::-1]+1)*127.5
        s_name = 'ep'+str(cur_ep)+'_sample'+str(cur_batch)+'_batch'+str(i)+'.png'
        cv.imwrite(save_dir+s_name,im_s)
    return True

def Save_load(ims,num,save_dir):
    b = np.shape(ims)[0]
    for i in range(b):
        im_S = ims[i,:,:,:]
        im_s = (im_S[:,:,::-1]+1)*127.5
        name = 'num_'+str(num)+'.png'
        cv.imwrite(save_dir+name,im_s)
    return True
