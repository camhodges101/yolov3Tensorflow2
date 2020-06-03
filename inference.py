from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, LeakyReLU, Add, Concatenate, Input, Softmax
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import yolonet, finaldetectionlayer
from utils import nms, showboxes
import wget

import time
'''
This first section loads our model graph and compiles the model.

Although Tensorflow 2.0 supports eager execution that isn't helpful here so we use a compiled graph like tensorflow 1.0
'''

InputPlaceholder=Input(shape=[608,608,3])

bbl,bbm,bbs=yolonet(InputPlaceholder)

modelout=(finaldetectionlayer(bbl,bbm,bbs))

mymodel=Model(inputs=InputPlaceholder, outputs=modelout)

mymodel.compile()

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 5
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 5
'''
Attempt to load the authors original weights file from the working directory, if not there then download from website. 
'''      
try:
  path='yolov3.weights'
  weight_reader = WeightReader(path)
except:

  wget(https://pjreddie.com/media/files/yolov3.weights)      
  path='yolov3.weights'
  weight_reader = WeightReader(path)

parameters=mymodel.variables
modelload={}
if True:
  '''
  This step creates a python dict, the keys are the layer numbers and each key returns a list of tensors, this includes the conv tensor and 4 batch norm components (beta, gamma, moving_mean and moving_variance)
  '''
  for para in parameters:
    layerref=((para.name).split("/")[0]).split("_")[-1]
    if layerref == "conv2d" or layerref == "normalization":
      layerref="0"
    if layerref not in modelload:
      modelload[layerref]=[]
    modelload[layerref]+=[para]

  for i in range(75):
    if i in [58,66,74]:
      #This section loads the weights for the final layers of each network branch, including bias values
      kernalshape=modelload[str(i)][0].shape
      Biassize=modelload[str(i)][1].shape
      Biasdata=weight_reader.read_bytes(Biassize[0])
      kernalshape=(kernalshape[3],kernalshape[2],kernalshape[1],kernalshape[0])

      kernaldata=weight_reader.read_bytes(np.product(kernalshape))
      # the darknet framework has a different order for its conv layers compared to tensorflow, darknet = [width,ch,height,batch] vs tensorflow= [batch, height, width, ch]
      modelload[str(i)][0].assign(np.transpose(kernaldata.reshape(kernalshape)),(2,3,1,0))
      modelload[str(i)][1].assign(Biasdata)

    else:
      #This section loads the weights for all layers except the final layers of each branch. 
      kernalshape=modelload[str(i)][0].shape
      kernalshape=(kernalshape[3],kernalshape[2],kernalshape[0],kernalshape[1])
      BNsize=(modelload[str(i)][1].shape)[0]
      beta=weight_reader.read_bytes(BNsize)
      gamma=weight_reader.read_bytes(BNsize)
      
      moving_mean=weight_reader.read_bytes(BNsize)
      moving_variance=weight_reader.read_bytes(BNsize)
      kernaldata=weight_reader.read_bytes(np.product(kernalshape))
      # the darknet framework has a different order for its conv layers compared to tensorflow, darknet = [width,ch,height,batch] vs tensorflow= [batch, height, width, ch]
      modelload[str(i)][0].assign(np.transpose(kernaldata.reshape(kernalshape),(2,3,1,0)))
      modelload[str(i)][1].assign(gamma)
      modelload[str(i)][2].assign(beta)
      modelload[str(i)][3].assign(moving_mean)
      modelload[str(i)][4].assign(moving_variance)

      



#Loads and runs inference on sample image person.jpg
out = np.array(Image.open('person.jpg').resize((608,608)))
ti=time.time()
detections=mymodel.predict(np.expand_dims(out,0).astype("float32"))
results=np.array(nms(detections))

showboxes(results[:,0],results[:,1],results[:,2],out)
print(time.time()-ti)