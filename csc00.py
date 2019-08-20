import cv2
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, Lambda, Dense
from keras.models import Model
from keras import backend as K


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis
  
def cv2_preprocess(prev,now):
  now=now[100:350, 0:639]
  prev=prev[100:350, 0:639]
  now=cv2.Canny(now,75,150)
  prev=cv2.Canny(prev,75,150)
  flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  return flow.astype('float32')

def image_tensor_func(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = cv2_preprocess(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

class CustomLayer( Layer ) :
    def call( self, xin )  :
        xout = tf.py_func( image_tensor_func, 
                           [xin],
                           'float32',
                           stateful=False,
                           name='cvOpt')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 250, 639, xin.shape[-1]] ) # explicitly set output shape
        return xout
    def compute_output_shape( self, sin ) :
        return ( sin[0], 250, 639, sin[-1] )



if __name__ == '__main__':

  # read the train targets
#  Y_train = np.genfromtxt('train.txt', delimiter=',')
#  np.save('Y_train',Y_train)
  Y_train = np.load('Y_train.npy')
  
  # plot the train targets
#  time_train = np.arange(0,np.floor(20400/20),0.05)
#  fig302, ax302 = plt.subplots()
#  ax302.plot(time_train,Y_train, label='spd_train')
  
  
  ## Read the video from specified path 
  #cam = cv2.VideoCapture("C:\\git\\speedchallenge\\data\\train.mp4") 
  #  
  #try: 
  #      
  #    # creating a folder named data 
  #    if not os.path.exists('data'): 
  #        os.makedirs('data') 
  #  
  ## if not created then raise error 
  #except OSError: 
  #    print ('Error: Creating directory of data') 
  #  
  ## frame 
  #currentframe = 0
  #  
  #while(True): 
  #      
  #    # reading from frame 
  #    ret,frame = cam.read() 
  #  
  #    if ret: 
  #        # if video is still left continue creating images 
  #        name = './data/frame' + str(currentframe) + '.jpg'
  #        print ('Creating...' + name) 
  #  
  #        # writing the extracted images 
  #        cv2.imwrite(name, frame) 
  #  
  #        # increasing counter so that it will 
  #        # show how many frames are created 
  #        currentframe += 1
  #    else: 
  #        break
  #  
  ## Release all space and windows once done 
  #cam.release() 
  #cv2.destroyAllWindows() 

# DEEVELOPING PREPROCESSING TECHNIQUE  
#  img = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1000.jpg',0) #C:\git\comma-speed-challenge\data
##  cv2.imshow('image',img)
#  imgc0 = img[100:350, 0:639]
##  cv2.imshow('image',imgc0)
#  
#  img1 = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1001.jpg',0) #C:\git\comma-speed-challenge\data
#  imgc1 = img1[100:350, 0:639]
##  cv2.imshow('image',imgc1)
#  
#  img2 = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1002.jpg',0) #C:\git\comma-speed-challenge\data
#  imgc2 = img2[100:350, 0:639]
##  cv2.imshow('image',imgc2)
#  
#  imgc12 = np.concatenate((imgc1, imgc2), axis=1)
##  cv2.imshow('image',imgc12)
#  edges12 = cv2.Canny(imgc12,75,150)
##  cv2.imshow('image',edges12)
#  
#  flow12 = cv2.calcOpticalFlowFarneback(imgc1, imgc2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
##  cv2.imshow('flow', draw_flow(imgc2, flow12))
#  
#  cny1 = cv2.Canny(imgc1,75,150)
#  cny2 = cv2.Canny(imgc2,75,150)
#  flowCny12 = cv2.calcOpticalFlowFarneback(cny1, cny2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
##  cv2.imshow('flow', draw_flow(cny2, flowCny12))
#  
#  flow_func_test = cv2_preprocess(img1,img2)
#  cv2.imshow('flow', draw_flow(imgc2, flow_func_test))
  
  x = Input(shape=(None,None,3))
  f = CustomLayer(name='custom')(x)
  c = Conv2D(1,(1,1), padding='same')(f)
  y = Dense(1,input_shape=(250,639))(c)
  
  model = Model( inputs=x, outputs=y )
  model.summary()
