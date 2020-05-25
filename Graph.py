from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Add, Concatenate, Input, Softmax
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import numpy as np



def yolonet(InputPlaceholder):
  def upscale(input_tensor):
    inputShape=tf.shape(input_tensor)
    x = tf.image.resize(input_tensor,
                        (inputShape[1]*2,inputShape[2]*2),
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                        preserve_aspect_ratio=False,
                        antialias=False,
                        name=None)
    return x

  class Conv_layer(tf.keras.Model):
    def __init__(self,filters,kernel_size,batchnorm,padding="SAME", stride=(1,1)):
      super(Conv_layer, self).__init__(name='')
      self.batchnorm=batchnorm
      self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size,stride,use_bias=not(batchnorm),padding=padding)
      self.strides=stride
      if (kernel_size==(3,3)):
        self.pads=1
      else:
        self.pads=0   
      #self.layername=(self.conv2a.name).split("/")[0]
      #print(self.layername)
      #try:
      self.bn2a = tf.keras.layers.BatchNormalization(momentum=0.9,epsilon=1e-5)#(name=self.layername)
      #except:
      #  self.bn2a = tf.keras.layers.BatchNormalization(name=self.layername+"_2")
      self.activation=LeakyReLU(alpha=0.1)
      
    def call(self, input_tensor):
      if self.strides==(2,2):
        x=tf.pad(input_tensor,tf.constant([[0,0],[self.pads,self.pads],[self.pads,self.pads],[0,0]]),mode="CONSTANT")
        
      else:
        x=input_tensor
      x = self.conv2a(x)
      
      if self.batchnorm:
        x = self.bn2a(x)

        x=self.activation(x)
      return x

  class residualblock(tf.keras.Model):
    def __init__(self,nb_filters,kernels):
      super(residualblock, self).__init__(name='')
      self.conv1=Conv_layer(nb_filters[0],kernels[0],batchnorm=True)
      self.conv2=Conv_layer(nb_filters[1],kernels[1],batchnorm=True)

    def call(self,input_tensor):
      x=self.conv1(input_tensor)
      x=self.conv2(x)
      x=x+input_tensor
      return x

  ##model=(Conv_layer(32,kernel_size=(3,3),batchnorm=False).call(x))
  class residuallayer(tf.keras.Model):
    def __init__(self,nb_filters,kernels, nb_blocks):
      super(residuallayer, self).__init__(name='')
      self.nb_blocks=nb_blocks
      self.nb_filters=nb_filters
      self.kernels=kernels
      self.conv1=Conv_layer(self.nb_filters[1],kernel_size=(3,3),padding="VALID",batchnorm=True,stride=(2,2))
      
    def call(self,input_tensor):
      
      x=self.conv1.call(input_tensor)
      
      for k in range(self.nb_blocks):
        x=residualblock(self.nb_filters,kernels=self.kernels).call(x)
      
      return x

  class darknet53(Model):
    def __init__(self):
      super(darknet53, self).__init__()
      self.conv1=Conv_layer(32,kernel_size=(3,3),batchnorm=True)
      self.resid1=residuallayer([32,64],[1,3],1)
      self.resid2=residuallayer([64,128],[1,3],2)
      self.resid3=residuallayer([128,256],[1,3],8)
      self.resid4=residuallayer([256,512],[1,3],8)
      self.resid5=residuallayer([512,1024],[1,3],4)

    def call(self,inputImage):
       
      x=self.conv1(inputImage)
      x1=self.resid1.call(x)
      x2=self.resid2.call(x1)
      x3=self.resid3.call(x2)
      x4=self.resid4.call(x3)
      x5=self.resid5.call(x4)
      return x3,x4,x5

  def darknet53function(testinput):
    x=Conv_layer(32,kernel_size=(3,3),batchnorm=True).call(testinput)
    x1=residuallayer([32,64],[1,3],1).call(x)
    x2=residuallayer([64,128],[1,3],2).call(x1)
    x3=residuallayer([128,256],[1,3],8).call(x2)
    x4=residuallayer([256,512],[1,3],8).call(x3)
    x5=residuallayer([512,1024],[1,3],4).call(x4)
    return x3,x4,x5    

  class lastLayers(Model):
    def __init__(self,nb_filters):
      super(lastLayers, self).__init__()
      self.nb_filters=nb_filters
      self.conv1=Conv_layer(self.nb_filters,kernel_size=(1,1),batchnorm=True)
      self.conv2=Conv_layer(self.nb_filters*2,kernel_size=(3,3),batchnorm=True)
      self.conv3=Conv_layer(self.nb_filters,kernel_size=(1,1),batchnorm=True)
      self.conv4=Conv_layer(self.nb_filters*2,kernel_size=(3,3),batchnorm=True)
      self.conv5=Conv_layer(self.nb_filters,kernel_size=(1,1),batchnorm=True)
    def call(self,inputImage):
      x=self.conv1.call(inputImage)
      x=self.conv2.call(x)
      x=self.conv3.call(x)
      x=self.conv4.call(x)
      x=self.conv5.call(x)
      return x


  class networkhead(Model):
    def __init__(self):
      pass
    def call(self, route1, route2, route3):
      super(networkhead, self).__init__()
      route=lastLayers(512).call(route3)
      bblBranch=Conv_layer(1024,(3,3),batchnorm=True).call(route)
      bblBranch=Conv_layer(255,(1,1),batchnorm=False).call(bblBranch)
      route=Conv_layer(256,(1,1),batchnorm=True).call(route)
      route=upscale(route)
      route=Concatenate(axis=-1)([route2,route])
      route=lastLayers(256).call(route)

      bbmBranch=Conv_layer(512,(3,3),batchnorm=True).call(route)
      bbmBranch=Conv_layer(255,(1,1),batchnorm=False).call(bbmBranch)
      route=Conv_layer(128,(1,1),batchnorm=True).call(route)
      route=upscale(route)
      route=Concatenate(axis=-1)([route1,route])

      route=lastLayers(128).call(route)

      bbsBranch=Conv_layer(256,(3,3),batchnorm=True).call(route)
      bbsBranch=Conv_layer(255,(1,1),batchnorm=False).call(bbsBranch)
      return bblBranch,bbmBranch,bbsBranch
  
  #route1, route2, route3 = darknet53().call(InputPlaceholder)
  route1, route2, route3 = darknet53function(InputPlaceholder)

  bblBranch,bbmBranch,bbsBranch = networkhead().call(route1,route2,route3)
  return bblBranch,bbmBranch,bbsBranch
 # print(bbsBranch.shape)

def finaldetectionlayer(bblBranch,bbmBranch,bbsBranch):
  def detectionlayer(Branch):
    numclasses=80

    GRID_W, GRID_H = Branch.shape[1],Branch.shape[1]
    cellsize=float(int(608/GRID_W))
    inputTensor=tf.reshape(Branch,[-1,GRID_W,GRID_H,3,5+numclasses])
    #print(inputTensor)
    anchors = np.array([[10,13],  [16,30],  [33,23],  [30,61],  [62,45],  [59,119],  [116,90],  [156,198],  [373,326]])
    anchRef=(GRID_H==19)*6+(GRID_H==38)*3+(GRID_H==76)*0

    anchors=anchors[anchRef:anchRef+3,:].reshape((1,1,1,3,2))

    cell_x = tf.dtypes.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)),tf.float32)
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))


    gridRef=tf.tile(tf.concat((cell_x,cell_y),axis=-1),[1,1,1,3,1])

    
    xy=(tf.sigmoid(inputTensor[...,0:2])+gridRef)*cellsize

    wh=tf.exp(inputTensor[...,2:4])*anchors
    #wh=anchors
    conf=tf.sigmoid(inputTensor[...,4:5])
    #conf=inputTensor[...,4:5]
    classProd=tf.sigmoid(inputTensor[...,5:])
    return tf.concat((xy,wh,conf,classProd),axis=-1)
  bblBranch=tf.reshape(detectionlayer(bblBranch),[-1,bblBranch.shape[1]*bblBranch.shape[2]*3,85])
  bbmBranch=tf.reshape(detectionlayer(bbmBranch),[-1,bbmBranch.shape[1]*bbmBranch.shape[2]*3,85])
  bbsBranch=tf.reshape(detectionlayer(bbsBranch),[-1,bbsBranch.shape[1]*bbsBranch.shape[2]*3,85])
  
  

  return tf.concat((bblBranch,bbmBranch,bbsBranch),axis=1)

