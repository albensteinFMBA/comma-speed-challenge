import cv2
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras

## read the train targets
#Y_train = np.genfromtxt('train.txt', delimiter=',')
#np.save('Y_train',Y_train)
#
## plot the train targets
#time_train = np.arange(0,np.floor(20400/20),0.05)
#fig302, ax302 = plt.subplots()
#ax302.plot(time_train,Y_train, label='spd_train')


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

img = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1000.jpg',0) #C:\git\comma-speed-challenge\data
imgc0 = img[0:350, 0:639]
cv2.imshow('image',imgc0)

img = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1001.jpg',0) #C:\git\comma-speed-challenge\data
imgc1 = img[0:350, 0:639]
cv2.imshow('image',imgc1)

img = cv2.imread('C:\\git\\comma-speed-challenge\\data\\frame1002.jpg',0) #C:\git\comma-speed-challenge\data
imgc2 = img[0:350, 0:639]
cv2.imshow('image',imgc2)

imgc12 = np.concatenate((imgc1, imgc2), axis=1)
cv2.imshow('image',imgc12)