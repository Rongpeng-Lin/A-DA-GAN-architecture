from utlis import *
from ops import *
import tensorflow as tf
import numpy as np
import cv2 as cv
import math,os,h5py

class DAE_GAN:
    def __init__(self,batch,epoch,im_size,hw_size,k,alpa,beta,im_dir,save_dir,saveS_dir,saveT_dir):
        self.batch = batch
        self.epoch = epoch
        self.im_size = im_size
        self.hw_size = hw_size
        self.positive_dir = [im_dir+'positive/'+name for name in os.listdir(im_dir+'positive')]
        self.negtive_dir = [im_dir+'negtive/'+name for name in os.listdir(im_dir+'negtive')]
        self.svhnMat = h5py.File(im_dir+'digitStruct.mat', mode='r')
        self.first = 64
        self.k = k
        self.alpa = alpa
        self.beta = beta
        self.one_ep_iter = One_ep_Iter(im_dir,batch)
        self.S = tf.placeholder(tf.float32,[None,im_size,im_size,3],'S')
        self.T = tf.placeholder(tf.float32,[None,im_size,im_size,3],'T')
        self.S_label = tf.placeholder(tf.float32,[None,10],'S_label')
        self.T_label = tf.placeholder(tf.float32,[None,10],'T_label')
        self.save_dir = save_dir
        self.saveS = saveS_dir
        self.saveT = saveT_dir

    def Encoder(self,x,reuse):
        with tf.variable_scope('encoder',reuse=reuse): 
            conv1 = conv('conv1',x,3*3,2,self.first,True)
            bn1 = BN('bn1',conv1)
            relu1 = lrelu('relu1',bn1)
            
            conv2 = conv('conv2',relu1,3*3,2,int(2*self.first),True)
            bn2 = BN('bn2',conv2)
            relu2 = lrelu('relu2',bn2)
            
            conv3 = conv('conv3',relu2,3*3,2,int(4*self.first),True)
            bn3 = BN('bn3',conv3)
            relu3 = lrelu('relu3',bn3)
            
            conv4 = conv('conv4',relu3,3*3,2,int(8*self.first),True)
            bn4 = BN('bn4',conv4)
            relu4 = lrelu('relu4',bn4)
            return relu4
         
    def F(self,features,k):
        with tf.variable_scope('f_location'):
            cha = [i.value for i in features.get_shape()][-1]
            
            conv1 = conv('conv1',features,3*3,2,int(2*cha),True)
            bn1 = BN('bn1',conv1)
            relu1 = lrelu('relu1',bn1)
            
            conv2 = conv('conv2',relu1,3*3,2,int(4*cha),True)
            bn2 = BN('bn2',conv2)
            relu2 = lrelu('relu2',bn2)
            
            out_xy = FC_location('fc1',relu2,2,self.im_size,self.hw_size)
            mask = get_mask(out_xy,self.hw_size,self.hw_size,self.im_size,k,self.batch)
            return mask
    
    def GAN_G(self,x,reuse):
        with tf.variable_scope('G_net',reuse=reuse):                        

            conv_trans1 = conv_trans('conv_trans1',x,5*5,1,256,True)       
            Lrelu1 =  lrelu('Lrelu1',conv_trans1)
            
            conv_trans2 = conv_trans('conv_trans2',Lrelu1,5*5,1,128,True)
            bn2 = BN('bn2',conv_trans2)
            Lrelu2 =  lrelu('Lrelu2',bn2)
            
            conv_trans3 = conv_trans('conv_trans3',Lrelu2,5*5,1,64,True)
            bn3 = BN('bn3',conv_trans3)
            Lrelu3 =  lrelu('Lrelu3',bn3)
            
            conv_trans4 = conv_trans('conv_trans4',Lrelu3,5*5,1,3,True)
            Lrelu4 =  tanh('Lrelu4',conv_trans4)
            return Lrelu4

    def GAN_D(self,name,x,reuse):
        with tf.variable_scope(name,reuse=reuse):
            conv1 = conv('conv1',x,3*3,2,64,True)
            bn1 = BN('bn1',conv1)
            relu1 = lrelu('relu1',bn1)
            
            conv2 = conv('conv2',relu1,3*3,2,128,True)
            bn2 = BN('bn2',conv2)
            relu2 = lrelu('relu2',bn2)
            
            conv3 = conv('conv3',relu2,3*3,2,256,True)
            bn3 = BN('bn3',conv3)
            relu3 = lrelu('relu3',bn3)
            
            conv4 = conv('conv4',relu3,3*3,2,512,True)
            bn4 = BN('bn4',conv4)
            relu4 = lrelu('relu4',bn4)
            
            D_out = FC('fc1',relu4,1)
            return D_out
    
    def DAE(self,x,reuse):
        with tf.variable_scope('DAE',reuse=reuse):
            encode_x = self.Encoder(x,reuse)
            mask = self.F(encode_x,self.k)
            x_mask = x*mask
            encode_mask_x = self.Encoder(x_mask,True)
                                             
            probs = resnet_classifer('classsifer',encode_mask_x)
            return encode_mask_x,probs

    def forward(self):
        self.s_DAE,self.s_out_label = self.DAE(self.S,False)
        self.t_DAE,self.t_out_label = self.DAE(self.T,True)
        self.s_pie = self.GAN_G(self.s_DAE,False)
        self.t_pie = self.GAN_G(self.t_DAE,True)
        
        self.t_D1 = self.GAN_D('D2',self.T,False)
        self.t_pie_D = self.GAN_D('D2',self.t_pie,True)
        
        self.t_D2 = self.GAN_D('D1',self.T,False)
        self.s_pie_D = self.GAN_D('D1',self.s_pie,True)
        
        self.s_pie_DAE,_ = self.DAE(self.s_pie,True)
        self.t_pie_DAE,_ = self.DAE(self.t_pie,True)
    
    def train(self):
        self.forward()
# Loss for G and DAE:
#         1、 Loss of s_DAE and s_pie_DAE:
        Lcst = tf.reduce_mean(tf.abs(self.s_DAE-self.s_pie_DAE))  # L1范数
#         2、 Loss of t_DAE and t_pie_DAE:
        Lsym = tf.reduce_mean(tf.abs(self.t_DAE-self.t_pie_DAE))  # L1范数
#         3、Make D2 judge t_pie as true:  
        loss_G_DAE1 = tf.reduce_mean(-1*tf.log(self.t_pie_D))
#         4、Make D1 judge s_pie as true:
        loss_G_DAE2 = tf.reduce_mean(-1*tf.log(self.s_pie_D))
#         5、Loss caused by classification: cross entropy, this alone corresponds to DAE:     
        cross_entroy = tf.reduce_mean(tf.reduce_mean((-1)*tf.log(self.s_out_label)*self.S_label + (-1)*tf.log(1-self.s_out_label)*(1-self.S_label),1))
        
        self.DAE_G_loss = self.alpa*Lcst + self.beta*Lsym + loss_G_DAE1 + loss_G_DAE2 + cross_entroy
        
        self.D1_loss = tf.reduce_mean((-1)*tf.log(self.t_D2) + (-1)*tf.log(1-self.s_pie_D))
        self.D2_loss = tf.reduce_mean((-1)*tf.log(self.t_D2) + (-1)*tf.log(1-self.t_pie_D))
        DAE_G_vars = [var for var in tf.trainable_variables() if 'DAE' in var.name or 'G_net' in var.name]
        D1_vars = [var for var in tf.trainable_variables() if 'D1' in var.name]
        D2_vars = [var for var in tf.trainable_variables() if 'D2' in var.name]
        optim_DAE_G = tf.train.AdamOptimizer().minimize(self.DAE_G_loss,var_list=DAE_G_vars)  
        optim_D1 = tf.train.AdamOptimizer().minimize(self.D1_loss,var_list=D1_vars) 
        optim_D2 = tf.train.AdamOptimizer().minimize(self.D2_loss,var_list=D2_vars) 
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter(self.save_dir,graph=sess.graph)
            Saver = tf.train.Saver(max_to_keep=20)
            
            savedir = self.save_dir+'model.ckpt'
            for i in range(self.epoch):
                for j in range(self.one_ep_iter):
                    ims_po,labels_po = get_im_label(j, self.batch, self.positive_dir, self.svhnMat)
                    ims_neg,labels_neg = get_im_label(j, self.batch, self.negtive_dir, self.svhnMat)
                    
                    fed_dict={self.S:ims_po,self.S_label:labels_po,self.T:ims_neg,self.T_label:labels_neg}
                    
                    _,LossD1 = sess.run([optim_D1,self.D1_loss],feed_dict=fed_dict)
                    print('LossD1: ',LossD1)
                    
                    _,LossD2 = sess.run([optim_D2,self.D2_loss],feed_dict=fed_dict)
                    print('LossD2: ',LossD2)
                     
                    _,S_sample,T_sample,LossDAE_G = sess.run([optim_DAE_G,self.s_pie,self.t_pie,self.DAE_G_loss],feed_dict=fed_dict)
                    print('LossDAE_G: ',LossDAE_G)
                    
                    save_im(S_sample, self.saveS, i, j, self.batch)
                    save_im(T_sample, self.saveT, i, j, self.batch)
                                        
                    step = int(i*self.epoch + j)
                    Saver.save(sess,savedir,global_step=step)
                    print('save_success at: ',step)
                    
    def load(self,load_dir,raw_im_dir,save_im_dir):
        self.forward()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter(self.save_dir,graph=sess.graph)
            Saver = tf.train.Saver()
            Saver.restore(sess,load_dir)
            for i,im_name in enumerate(os.listdir(raw_im_dir)):
                im_bgr = cv.resize(cv.imread(raw_im_dir+im_name),(64,64),interpolation=cv.INTER_CUBIC)
                im_rgb_unpro = im_bgr[:,:,::-1]
                im_rgb = np.expand_dims(((im_rgb_unpro/255)-0.5)*2,0)
#             fed_dict={self.S:ims_po,self.S_label:labels_po}
                fed_dict={self.S:im_rgb}
                s_tar = sess.run(self.s_pie,feed_dict=fed_dict)
                Save_load(s_tar,i,save_im_dir)
                print('save at: ',i)
