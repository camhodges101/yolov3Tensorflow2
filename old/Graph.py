#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 05:19:11 2019

@author: cameron
"""
import os
import numpy as np
import random
import tensorflow as tf
from PIL import Image

tf.reset_default_graph()
x1=tf.placeholder('float',[None,416,416,3])

def conv2d(x,W,wd,name):
    return tf.nn.conv2d(x, W, strides=[1,wd[0],wd[1],1], padding='SAME',name=name)

def lrelu(x,alpha):
    return tf.nn.leaky_relu(x,alpha,name=None)

def maxpool2d(x,wd):
    return tf.nn.max_pool(x,ksize=(1,2,2,1), strides=[1,wd[0],wd[1],1], padding='VALID',name=None)
	
def batnorm(x,name):
	return tf.layers.batch_normalization(x,name=name)
sd=0.03
weights={'W_conv1':tf.Variable(tf.random_normal([3,3,3,32], mean=0, stddev=0.3),name="D_W_conv",trainable=False),
             'W_conv2':tf.Variable(tf.random_normal([3,3,32,64], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv3':tf.Variable(tf.random_normal([3,3,64,128], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv4':tf.Variable(tf.random_normal([1,1,128,64], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv5':tf.Variable(tf.random_normal([3,3,64,128], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv6':tf.Variable(tf.random_normal([3,3,128,256], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv7':tf.Variable(tf.random_normal([1,1,256,128], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv8':tf.Variable(tf.random_normal([3,3,128,256], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv9':tf.Variable(tf.random_normal([3,3,256,512], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv10':tf.Variable(tf.random_normal([1,1,512,256], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv11':tf.Variable(tf.random_normal([3,3,256,512], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv12':tf.Variable(tf.random_normal([1,1,512,256], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv13':tf.Variable(tf.random_normal([3,3,256,512], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv14':tf.Variable(tf.random_normal([3,3,512,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv15':tf.Variable(tf.random_normal([1,1,1024,512], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv16':tf.Variable(tf.random_normal([3,3,512,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv17':tf.Variable(tf.random_normal([1,1,1024,512], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv18':tf.Variable(tf.random_normal([3,3,512,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv19':tf.Variable(tf.random_normal([3,3,1024,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv20':tf.Variable(tf.random_normal([3,3,1024,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv21':tf.Variable(tf.random_normal([1,1,512,64], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv22':tf.Variable(tf.random_normal([3,3,1280,1024], mean=0, stddev=0.003),name="D_W_conv",trainable=False),
             'W_conv23':tf.Variable(tf.random_normal([1,1,1024,425], mean=0, stddev=0.003),name="D_W_conv",trainable=False)}
             
#saver_W=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="D_"))
#saver_Wout=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="bottom"))
biases={'B_conv23':tf.Variable(tf.ones([425]),name="D_conv")}
saver_W=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="D_"))

#Network Definitions

weightsList=['W_conv1',
             'W_conv2',
             'W_conv3',
             'W_conv4',
             'W_conv5',
             'W_conv6',
             'W_conv7',
             'W_conv8',
             'W_conv9',
             'W_conv10',
             'W_conv11',
             'W_conv12',
             'W_conv13',
             'W_conv14',
             'W_conv15',
             'W_conv16',
             'W_conv17',
             'W_conv18',
             'W_conv19',
             'W_conv20',
             'W_conv21',
             'W_conv22',
             'W_conv23']


#Conv1 7x7x64 s-2
#Maxpool 2x2s2
#conv2 3x3x192s1
#maxpool 2x2s2
#conv3 1x1x128 s1
#conv4 3x3x256 s1
#conv5 1x1x256 s1
#conv6 3x3x512 s1
#maxpool 2x2s2
#conv7 1x1x256 s1
#conv8 3x3x512 s1
#conv9 1x1x256 s1
#conv10 3x3x512 s1
#conv11 1x1x256 s1
#conv12 3x3x512 s1
#conv13 1x1x256 s1
#conv14 3x3x512 s1
#conv15 1x1x512 s1
#conv16 3x3x1024 s1
#maxpool 2x2 s2
#conv17 1x1x512 s1
#conv18 3x3x1024 s1
#conv19 1x1x512 s1
#conv20 3x3x1024 s1
#conv21 3x3x1024 s1
#conv22 3x3x1024 s2
#conv23 3x3x1024
#conv24 3x3x1024
def YoloNet(x):
    conv1=conv2d(x,weights['W_conv1'],(1,1),'L1')
    conv1=lrelu(conv1,0.1)
    conv1=maxpool2d(conv1,(2,2))
    conv1=batnorm(conv1,'norm01')
    
    
    conv2=conv2d(conv1,weights['W_conv2'],(1,1),'L2')
    conv2=lrelu(conv2,0.1)
    conv2=maxpool2d(conv2,(2,2))
    conv2=batnorm(conv2,'norm02')
    
    conv3=conv2d(conv2,weights['W_conv3'],(1,1),'L3')
    conv3=lrelu(conv3,0.1)
    conv3=batnorm(conv3,'norm03')
    
    conv4=conv2d(conv3,weights['W_conv4'],(1,1),'L4')
    conv4=lrelu(conv4,0.1)
    conv4=batnorm(conv4,'norm04')
    
    conv5=conv2d(conv4,weights['W_conv5'],(1,1),'L5')
    conv5=lrelu(conv5,0.1)
    conv5=maxpool2d(conv5,(2,2))
    conv5=batnorm(conv5,'norm05')
    
    conv6=conv2d(conv5,weights['W_conv6'],(1,1),'L6')
    conv6=lrelu(conv6,0.1)
    conv6=batnorm(conv6,'norm06')
    
    conv7=conv2d(conv6,weights['W_conv7'],(1,1),'L7')
    conv7=lrelu(conv7,0.1)
    conv7=batnorm(conv7,'norm07')
    
    conv8=conv2d(conv7,weights['W_conv8'],(1,1),'L8')
    conv8=lrelu(conv8,0.1)
    conv8=maxpool2d(conv8,(2,2))
    conv8=batnorm(conv8,'norm08')
    
    conv9=conv2d(conv8,weights['W_conv9'],(1,1),'L9')
    conv9=lrelu(conv9,0.1)
    conv9=batnorm(conv9,'norm09')
    
    conv10=conv2d(conv9,weights['W_conv10'],(1,1),'L10')
    conv10=lrelu(conv10,0.1)
    conv10=batnorm(conv10,'norm10')
    
    conv11=conv2d(conv10,weights['W_conv11'],(1,1),'L11')
    conv11=lrelu(conv11,0.1)
    conv11=batnorm(conv11,'norm11')
    
    conv12=conv2d(conv11,weights['W_conv12'],(1,1),'L12')
    conv12=lrelu(conv12,0.1)
    conv12=batnorm(conv12,'norm12')
    
    conv13=conv2d(conv12,weights['W_conv13'],(1,1),'L13')
    conv13=lrelu(conv13,0.1)
    skip=conv13

    conv13=maxpool2d(conv13,(2,2))
    conv13=batnorm(conv13,'norm13')
    
    conv14=conv2d(conv13,weights['W_conv14'],(1,1),'L14')
    conv14=lrelu(conv14,0.1)
    conv14=batnorm(conv14,'norm14')
    
    conv15=conv2d(conv14,weights['W_conv15'],(1,1),'L15')
    conv15=lrelu(conv15,0.1)
    conv15=batnorm(conv15,'norm15')
    
    conv16=conv2d(conv15,weights['W_conv16'],(1,1),'L16')
    conv16=lrelu(conv16,0.1)
    conv16=batnorm(conv16,'norm16')
    
    conv17=conv2d(conv16,weights['W_conv17'],(1,1),'L17')
    conv17=lrelu(conv17,0.1)
    conv17=batnorm(conv17,'norm17')
    
    conv18=conv2d(conv17,weights['W_conv18'],(1,1),'L18')
    conv18=lrelu(conv18,0.1)
    conv18=batnorm(conv18,'norm18')
    #conv18=tf.concat((conv18,skip),axis=3)
    
    conv19=conv2d(conv18,weights['W_conv19'],(1,1),'L19')
    conv19=lrelu(conv19,0.1)
    conv19=batnorm(conv19,'norm19')
    
    conv20=conv2d(conv19,weights['W_conv20'],(1,1),'L19')
    conv20=lrelu(conv20,0.1)
    conv20=batnorm(conv20,'norm20')

    conv21=conv2d(skip,weights['W_conv21'],(1,1),'L19')
    conv21=lrelu(conv21,0.1)
    conv21=batnorm(conv21,'norm21')
    
    conv21=tf.space_to_depth(conv21, block_size=2)

    conv21=tf.concat((conv21,conv20),axis=3)
    
    conv22=conv2d(conv21,weights['W_conv22'],(1,1),'L19')
    conv22=lrelu(conv22,0.1)
    conv22=batnorm(conv22,'norm22')
    
    conv23=tf.nn.bias_add(conv2d(conv22,weights['W_conv23'],(1,1),'L20'),biases['B_conv23'])

    
    
    
        #tf.Print(convfc2)
    return conv23  

def finalLayer(x):
    prObj=tf.expand_dims(tf.sigmoid(x[...,4]),-1)
    XYpara=tf.sigmoid(x[...,0:2])
    anchors=[0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
    
    anchor_conf=np.reshape(anchors, [1,1,1,5,2]).astype("float32")
    
    whpara=tf.multiply(anchor_conf,tf.exp(x[...,2:4]))/32
    #whpara=tf.exp(x[...,2:4])
    
    GRID_W, GRID_H = 13,13
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [1, 1, 1, 5, 1])
    XYpara=tf.add(XYpara,cell_grid)
    classPara=x[...,5:]
    return tf.concat((XYpara,whpara,prObj,classPara),axis=4)

network=YoloNet(x1)
network=tf.reshape(network,[-1,13,13,5,85])
network=finalLayer(network)    

saver_norm=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="norm"))
#%%

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4
        
        
direct="/home/cameron/Downloads/"

listing=os.listdir(direct)


wt_path=direct+"/"+listing[1]
weight_reader = WeightReader(wt_path)

weight_reader.reset()
nb_conv = 23
imgpath='/home/cameron/data_sets/objectDetection/VOC2012/JPEGImages/2011_003912.jpg'
data=Image.open(imgpath)
img=(np.array(data.resize((416,416)))).reshape((1,416,416,3))


import matplotlib.pyplot as plt
#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())


    for a in range(1,23):
        variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="norm"+f"{a:02d}")
        size=(variables[0].shape)[0].value
    
        beta  = weight_reader.read_bytes(size)
        gamma = weight_reader.read_bytes(size)
        mean  = weight_reader.read_bytes(size)
        var   = weight_reader.read_bytes(size)
    
        variables[0].load(gamma,session=sess)
        variables[1].load(beta,session=sess)
        variables[2].load(mean,session=sess)
        variables[3].load(var,session=sess)
        
        wt_tensor=weights[weightsList[a-1]]
        conv_shape=((wt_tensor.shape)[0].value,(wt_tensor.shape)[1].value,(wt_tensor.shape)[2].value,(wt_tensor.shape)[3].value)
        kernel = weight_reader.read_bytes(np.prod(conv_shape))#shape=(864,)
        kernel = kernel.reshape(list(reversed(conv_shape)))#shape=(32,3,3,3)
        kernel = kernel.transpose([2,3,1,0])#shape=(3,3,3,32)
        weights[weightsList[a-1]].load(kernel, session=sess)#Loads weights to tensor
        
    wt_tensor=weights["W_conv23"]
    conv_shape=((wt_tensor.shape)[0].value,(wt_tensor.shape)[1].value,(wt_tensor.shape)[2].value,(wt_tensor.shape)[3].value)
    bs_tensor=biases["B_conv23"]
    bias   = weight_reader.read_bytes(bs_tensor.shape[0].value)
    kernel = weight_reader.read_bytes(np.prod(conv_shape))
    kernel = kernel.reshape(list(reversed(conv_shape)))
    kernel = kernel.transpose([2,3,1,0])
    wt_tensor.load(kernel,session=sess)
    bs_tensor.load(bias,session=sess)
    #save_path_W=saver_W.save(sess, 'pretrained_weights/yoloV2_WB.ckpt',write_meta_graph=False)
    #save_path_norm=saver_norm.save(sess, 'pretrained_weights/yoloV2_norm.ckpt',write_meta_graph=False)
    output=sess.run(network,feed_dict={x1:img})#.reshape(13,13,5,85)
    #print(np.array(weight_reader.read_bytes(555555555555555555555555555555)).shape)    
    print(np.argmax(output[0,7,6,4,5:]))
    '''
    array([ 6.997652  ,  7.920007  , 59.53056   , 10.271783  ,  0.55847234],
    
    '''

    #print(sess.run(variables))