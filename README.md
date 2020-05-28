# yolov3

This is not my original model, I've taken the Yolov3 model found  here (https://pjreddie.com/darknet/yolo/) and reimplimented the original authors model (including weights) to the Tensorflow 2.0 model. Theres plenty of examples of this on Github. 

The purpose of this was to learn more about how the Yolo models work, get familar with the changes from TF 1.0 to TF 2.0 and create a baseline model for some transfer learning projects I'm working on.

The model is almost identical to the authors original model with the exception of the source image padding, the authors original model keeps the image aspect ratio the same and pads the image with zeros to achieve the 608 x 608 input size. In this implementation I just resize the image to 608 x 608.
In my limited experiments this seems to give better performance on very wide angle input images. 

Tasks still to do 

1. create post processing step to resize images and detection boxes to original image size
2. implement final training process for transfer learning
 