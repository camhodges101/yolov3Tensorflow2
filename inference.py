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
from google.colab import drive
drive.mount('/content/drive')       
try:
  path='drive/My Drive/Colab Notebooks/yolov3/yolov3.weights'
  weight_reader = WeightReader(path)
except:

  ! wget https://pjreddie.com/media/files/yolov3.weights        
  path='yolov3.weights'
  weight_reader = WeightReader(path)

parameters=mymodel.variables
modelload={}
if True:
  for para in parameters:
    layerref=((para.name).split("/")[0]).split("_")[-1]
    if layerref == "conv2d" or layerref == "normalization":
      layerref="0"
    if layerref not in modelload:
      modelload[layerref]=[]
    modelload[layerref]+=[para]

  for i in range(75):
    if i in [58,66,74]:
      kernalshape=modelload[str(i)][0].shape
      Biassize=modelload[str(i)][1].shape
      Biasdata=weight_reader.read_bytes(Biassize[0])
      kernalshape=(kernalshape[3],kernalshape[2],kernalshape[1],kernalshape[0])

      kernaldata=weight_reader.read_bytes(np.product(kernalshape))
      modelload[str(i)][0].assign(np.transpose(kernaldata.reshape(kernalshape)),(2,3,1,0))
      modelload[str(i)][1].assign(Biasdata)

    else:
      kernalshape=modelload[str(i)][0].shape
      kernalshape=(kernalshape[3],kernalshape[2],kernalshape[0],kernalshape[1])
      BNsize=(modelload[str(i)][1].shape)[0]
      beta=weight_reader.read_bytes(BNsize)
      gamma=weight_reader.read_bytes(BNsize)
      
      moving_mean=weight_reader.read_bytes(BNsize)
      moving_variance=weight_reader.read_bytes(BNsize)
      kernaldata=weight_reader.read_bytes(np.product(kernalshape))
      modelload[str(i)][0].assign(np.transpose(kernaldata.reshape(kernalshape),(2,3,1,0)))
      modelload[str(i)][1].assign(gamma)
      modelload[str(i)][2].assign(beta)
      modelload[str(i)][3].assign(moving_mean)
      modelload[str(i)][4].assign(moving_variance)

      
