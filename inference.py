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