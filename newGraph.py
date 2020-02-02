# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 09:25:29 2019

@author: hodgec
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
#%%

        
with tf.name_scope('weights'):
    weights={}
    biases={}
def conv2d(x,W,wd,name,padding="SAME"):
    x=tf.nn.conv2d(x, W, strides=[1,wd[0],wd[1],1], padding=padding,name=name)
    return x
def lrelu(x,alpha):
    return tf.nn.leaky_relu(x,alpha,name=None)

def batnorm(x,name):
    with tf.name_scope('batchNormparas'):
    	return tf.layers.batch_normalization(x,name=name)

tf.reset_default_graph()
def convlayer(x,nb_filters,kernal_size,name,layerID,padding,stride=1,batchnorm=True):
    with tf.name_scope(name):
        nb_inputfilters=x.shape[3].value
        #print(layerID,"-",nb_inputfilters,"-",nb_filters)
        weightname,normname='conv'+str(f"{layerID:02d}"),'norm'+str(f"{layerID:02d}")
        weights[weightname]=tf.Variable(tf.random_normal([kernal_size,kernal_size,nb_inputfilters,nb_filters], mean=0, stddev=0.3),name=name,trainable=False)
        x=conv2d(x,weights[weightname],(stride,stride),'L'+str(f"{layerID:02d}"),padding=padding)
        x=lrelu(x,0.1)
        
        if batchnorm:
            x=batnorm(x,normname)
        
        else:
            biases['bias'+str(f"{layerID:02d}")]=tf.Variable(tf.zeros([x.shape[-1].value]),name="ConvBias"+str(f"{layerID:02d}"))
            x=tf.add(x,biases['bias'+str(f"{layerID:02d}")])
        layerID+=1
        return layerID,x
    
def residuallayer(x,layerID,nb_filters,kernal_size,nb_blocks):
    x=tf.pad(x,tf.constant([[0,0],[1,1],[1,1],[0,0]]))
    layerID,x=convlayer(x,nb_filters[1],kernal_size[1],"ConvL"+str(layerID),layerID,padding="VALID",stride=2)
    xskip=x
    for k in range(nb_blocks):
        layerID,x=convlayer(x,nb_filters[0],kernal_size[0],"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
        layerID,x=convlayer(x,nb_filters[1],kernal_size[1],"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    print(xskip.shape,x.shape)
    xp=tf.add(xskip,x)
    return layerID,xp
   
	


	
##############################################    

def darknet53(x,layerID=1):
    layerID,x1=convlayer(x,32,3,"ConvL1",layerID,padding="SAME",stride=1)
    layerID,x2=residuallayer(x1,layerID,[32,64],[1,3],1)
    layerID,x3=residuallayer(x2,layerID,[64,128],[1,3],2)
    layerID,x4=residuallayer(x3,layerID,[128,256],[1,3],8)
    layerID,x5=residuallayer(x4,layerID,[256,512],[1,3],8)
    print("Broken Layer - ",layerID)
    layerID,x6=residuallayer(x5,layerID,[512,1024],[1,3],4)
    
    return layerID,x4,x5,x6

def upscale(input_data):
    input_shape = tf.shape(input_data)
    output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    return output

#Now we have 3 tensors (?,32,32,256) - (?,16,16,512) - (?,8,8,1024)

def lastLayer(x,nb_filters,layerID):
    
    
    layerID,x=convlayer(x,nb_filters,1,"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    
    layerID,x=convlayer(x,nb_filters*2,3,"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    
    layerID,x=convlayer(x,nb_filters,1,"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    
    layerID,x=convlayer(x,nb_filters*2,3,"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    
    layerID,x=convlayer(x,nb_filters,1,"ConvL"+str(layerID),layerID,padding="SAME",stride=1)
    return layerID,x
#    layerID,x=convlayer(x,nb_filters*2,3,"L"+str(layerID),layerID,stride=1)
#    weightname,normname='conv'+str(f"{layerID:02d}"),'norm'+str(f"{layerID:02d}")
#    nb_inputfilters=x.shape[3].value
#    weights[weightname]=tf.Variable(tf.random_normal([1,1,nb_inputfilters,out_filters], mean=0, stddev=0.3),name="L"+str(layerID),trainable=False)
#    #print(x,"-",weightname,"-",layerID)
#    x=conv2d(x,weights[weightname],(1,1),"L"+str(layerID))
#
def yolonetwork(x):    
    with tf.name_scope("Yolograph"): 
    
        layerID, route1, route2, route3 = darknet53(x)
        
        layerID,route=lastLayer(route3,512,layerID)
        layerID,bblBranch=convlayer(route,1024,3,"ConvL"+str(layerID),layerID,padding="SAME")
        layerID,bblBranch=convlayer(bblBranch,255,1,"ConvL"+str(layerID),layerID,padding="SAME",batchnorm=False)
        layerID,route=convlayer(route,256,1,"ConvL"+str(layerID),layerID,padding="SAME")
        route=upscale(route)
        route=tf.concat((route2,route),axis=-1)
        layerID,route=lastLayer(route,256,layerID)
        
        layerID,bbmBranch=convlayer(route,512,3,"ConvL"+str(layerID),layerID,padding="SAME")
        layerID,bbmBranch=convlayer(bbmBranch,255,1,"ConvL"+str(layerID),layerID,padding="SAME",batchnorm=False)
        layerID,route=convlayer(route,128,1,"ConvL"+str(layerID),layerID,padding="SAME")
        route=upscale(route)
        route=tf.concat((route1,route),axis=-1)
        layerID,route=lastLayer(route,128,layerID)
        layerID,bbsBranch=convlayer(route,256,3,"ConvL"+str(layerID),layerID,padding="SAME")
        layerID,bbsBranch=convlayer(bbsBranch,255,1,"ConvL"+str(layerID),layerID,padding="SAME",batchnorm=False)
        #layerID,bbm=lastLayer(route2,512,3*85,layerID)
        #layerID,bbs=lastLayer(route3,1024,3*85,layerID)
        ####Need to find a way to upscale to 13x13, 26x26 and 52x52 
        ##############################################
        def detection_layer(Branch):
            numclasses=80
            GRID_W, GRID_H = Branch.shape[1].value,Branch.shape[1].value
            inputTensor=tf.reshape(Branch,[-1,GRID_W,GRID_H,3,5+numclasses])
            
            anchors = np.array([[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]])
            anchRef=(GRID_H==19)*6+(GRID_H==38)*3+(GRID_H==76)*0
            
            anchors=anchors[anchRef:anchRef+3,:].reshape((1,1,1,3,2))
            
            cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
            cell_y = tf.transpose(cell_x, (0,2,1,3,4))
            gridRef=tf.tile(tf.concat((cell_x,cell_y),axis=-1),[1,1,1,3,1])
            xy=tf.add(tf.sigmoid(inputTensor[...,0:2]),gridRef)
            
            wh=tf.multiply(tf.exp(inputTensor[...,2:4]),anchors)
            
            conf=tf.nn.sigmoid(inputTensor[...,4:5])
            #conf=inputTensor[...,4:5]
            classProd=tf.nn.sigmoid(inputTensor[...,5:])
            return tf.concat((xy,wh,conf,classProd),axis=-1)
        
        bblBranch=tf.reshape(detection_layer(bblBranch),[1,-1,5+80])
        bbmBranch=tf.reshape(detection_layer(bbmBranch),[1,-1,5+80])
        bbsBranch=tf.reshape(detection_layer(bbsBranch),[1,-1,5+80])
        return route3
        #return tf.concat((bblBranch,bbmBranch,bbsBranch),axis=1)
with tf.name_scope("inputImage"):
    x1=tf.placeholder('float',[1,608,608,3])

detectionnetwork=yolonetwork(x1)
#%%#############################################
#import tensorflow as tf

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 5
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 5
        
        
path='yolov3.weights'


weight_reader = WeightReader(path)
#saver_variables=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

def ConvertWeights():
    with tf.name_scope("weightsConvert"):
        saver_variables=tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        with tf.Session() as sess:
            variables=tf.global_variables()
            weight_reader.reset()
            for varRef in range(len(variables)):
                if variables[varRef].op.name.split("/")[-1:][0][:5] == "ConvL":
                    #print(varRef,"-",variables[varRef+1].op.name)    
                    if variables[varRef+1].op.name[:4] == "norm":
                        #print(varRef,"-",variables[varRef+1].op.name)
                        print("Loading Conv and Batch Norm Weights "+str(variables[varRef].op.name))
                        filtersize=variables[varRef].shape[-1].value
                        gamma=weight_reader.read_bytes(filtersize)
                        beta=weight_reader.read_bytes(filtersize)
                        moving_mean=weight_reader.read_bytes(filtersize)
                        moving_average=weight_reader.read_bytes(filtersize)
                        convshape=(variables[varRef].shape[3].value,variables[varRef].shape[2].value,variables[varRef].shape[0].value,variables[varRef].shape[1].value)
                        convweights=weight_reader.read_bytes(np.prod(convshape))
                        variables[varRef+1].load(gamma,session=sess)
                        variables[varRef+2].load(beta,session=sess)
                        variables[varRef+3].load(moving_mean,session=sess)
                        variables[varRef+4].load(moving_average,session=sess)
                        convweights = convweights.reshape(convshape)#shape=(32,3,3,3)
                        convweights = np.transpose(convweights,(2,3,1,0))#shape=(3,3,3,32)
                        variables[varRef].load(convweights,session=sess)
                    elif variables[varRef+1].op.name.split("/")[-1:][0][:5] == "ConvB":
                        print("Loading Conv and Bias Weights "+str(variables[varRef].op.name))
                        filtersize=variables[varRef].shape[-1].value
                        biasweights=weight_reader.read_bytes(filtersize)
                        convshape=(variables[varRef].shape[3].value,variables[varRef].shape[2].value,variables[varRef].shape[0].value,variables[varRef].shape[1].value)
                        convweights=weight_reader.read_bytes(np.prod(convshape))
                        variables[varRef+1].load(biasweights,session=sess)
                        convweights = convweights.reshape(convshape)#shape=(32,3,3,3)
                        convweights = convweights.transpose([2,3,1,0])#shape=(3,3,3,32)
                        variables[varRef].load(convweights,session=sess)
            save_path_W=saver_variables.save(sess,'checkpoint/yolov3_TF.ckpt',write_meta_graph=False)
#ConvertWeights()
#%%#############################################

#%%#############################################
#file='dog.jpg'
#0.012264341

file='HF_Baidu_421.png'
img=Image.open(file)
width,height=img.size
factor=608/width
img=np.array(img.resize((608,608)))/255
#img=img.resize((608,int(factor*height)))
#padding=np.zeros((int((608-img.size[1])/2),608,3))
#img=np.concatenate((padding,np.array(img)/255,padding),axis=0)
img=np.expand_dims(img,axis=0)
#%%#############################################
def inference():
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        #ConvertWeights()
        writer=tf.summary.FileWriter("/home/cameron/Documents/tboard")
        writer.add_graph(sess.graph)
        #saver_variables.restore(sess,'checkboard/yolov3_TF.ckpt')
        detections=sess.run(detectionnetwork,feed_dict={x1:img})
        #print("no weights")
        print(img.shape)
        print(np.min(img),"-",np.max(img))
        print(detections.shape)
        print((np.array(detections)))
    #variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="norm"+f"{a:02d}")
    
        #tf.Print(convfc2)
inference()
#%%


#%%
