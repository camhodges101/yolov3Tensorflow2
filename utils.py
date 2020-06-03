#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:48:16 2020

@author: cameron
"""



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




def calc_iou_batch(pred_boxes, gt_box):
    '''
    This function takes a single ground truth or comparison box and compares to a list of predicted boxes. 
    
    In the case of inference the GT box is the predicted box with the highest confidence. 
    '''
    results=[]
    #switch input from xywh, to xmin,ymin, xmax,ymax
    gtx,gty,gtw,gth = gt_box
    x1_t, y1_t, x2_t, y2_t = gtx-0.5*gtw,gty-0.5*gth,gtx+0.5*gtw,gty+0.5*gth
    for box in pred_boxes:
     
  
      x,y,w,h = box
      
      x1_p, y1_p, x2_p, y2_p = x-0.5*w,y-0.5*h,x+0.5*w,y+0.5*h
      if (x1_p > x2_p) or (y1_p > y2_p):
          raise AssertionError(
              "Prediction box is malformed? pred box: {}".format(box))
      if (x1_t > x2_t) or (y1_t > y2_t):
          raise AssertionError(
              "Ground Truth box is malformed? true box: {}".format(gt_box))
  
      if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
          iou= 0.0
      else:
        far_x = np.min([x2_t, x2_p])
        near_x = np.max([x1_t, x1_p])
        far_y = np.min([y2_t, y2_p])
        near_y = np.max([y1_t, y1_p])
    
        inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
        true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
        pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
        iou = inter_area / (true_box_area + pred_box_area - inter_area)
      results+=[iou]
    return results



def nms(detections,confthrs=0.5,iouthrs=0.4):  
  '''
  This is a simple class based non max suppression function, 
  
  This step removes multiple duplicate bounding box predictions from the same object
  
  In future this could be replaced by the NMS function from the tensorflow library if it proves to be more efficient. 
  
  https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
  '''
  
  bbox=detections[0,:,0:4]
  conf=detections[0,:,4]
  
  classconf=detections[0,:,5:]
  confmask=conf>confthrs
  bbox=bbox[np.nonzero(confmask)]
  conf=conf[np.nonzero(confmask)]
  classconf=classconf[np.nonzero(confmask)]
  bbox=bbox[(-conf).argsort()]
  classconf=classconf[(-conf).argsort()]
  conf=conf[(-conf).argsort()]
  count=0

  count=0
  results=[]
  uniqueclasses = list(set(np.argmax(classconf,axis=-1)))
  for class_ in uniqueclasses:

    
    clsmask=np.nonzero(np.argmax(classconf,axis=1)==class_)
    
    classdetection=conf[clsmask]
    classboxes=bbox[clsmask]
    classCls=classconf[clsmask]
    
    while len(classdetection)>0:
      
      spdetection=classdetection[0]

      spbbox=classboxes[0]

      spclass=np.argmax(classCls[0])
      results.append([spbbox,spdetection,spclass])
      
      ious = np.array(calc_iou_batch(classboxes[1:],spbbox))
      ioumask=np.nonzero(ious < iouthrs)
      tempclassdetection=classdetection[1:]
      tempclassboxes=classboxes[1:]
      tempclassCls=classCls[1:]
      classdetection=tempclassdetection[ioumask]
      classboxes=tempclassboxes[ioumask]
      classCls=tempclassCls[ioumask]

  return results





def showboxes(X,scores,classes,img):
  #img=np.array(Image.open(file).resize((608,608)))/255
  def drawconfidence(img,score,bb1,class_,colorcode):
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (bb1[0],bb1[1]-2)
            fontScale              = 0.75
            fontColor              = colorcode
            lineType               = 2
            text                   = str(int(score*100))+"% "+class_

            cv2.putText(img,text, 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)
            return img
  for box,score,class_ in zip(X,scores,classes):
    #print(box)
    colordict={'horse':(0,1,0),'dog':(1,0,0),'person':(0,0,1),'bus':(1,0,1)}
    classname=labeldict[str(class_)]
    try:
      colorcode= colordict[classname]
    except:
      colorcode=(0,1,0)
    xmin = int(box[0]-0.5*box[2])
    ymin = int(box[1]-0.5*box[3])
    xmax = int(box[0]+0.5*box[2])
    ymax = int(box[1]+0.5*box[3])
    img = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), colorcode, 2)
    img = drawconfidence(img,score,(xmin,ymin),classname,colorcode)
  plt.imshow(img)
  plt.show()

#results=np.array(results)

