import cv2
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Layer, Input, Conv2D, Lambda, Dense
from keras.models import Model
from keras import backend as K

def create_dframe():
  #images = [f for f in os.listdir('C:/git/comma-speed-challenge/data') if os.path.isfile(os.path.join('C:/git/comma-speed-challenge/data', f))]
  # that line above seems to not get a list of sequential file names, which is what i expected. 
  # i know the file names, so i can just write the paths explcitly.
  path_prev = []
  path_now = []
  spd = np.load('Y_train.npy')
  spd_now = []
  for i in np.arange(20400-1):
    path_prev.append('C:\\git\\comma-speed-challenge\\data\\frame' + str(i) + '.jpg')
    path_now.append('C:\\git\\comma-speed-challenge\\data\\frame' + str(i+1) + '.jpg')
    spd_now.append(np.mean(np.array(spd[i],spd[i+1])))
  d={'path_prev':path_prev,'path_now':path_now,'spd':spd_now}
  df = pd.DataFrame(d,columns=['path_prev','path_now','spd'])
  return df

def batch_shuffle(dframe):
  """
  Randomly shuffle pairs of rows in the dataframe, separates train and validation data
  generates a uniform random variable 0->9, gives 20% chance to append to valid data, otherwise train_data
  return tuple (train_data, valid_data) dataframes
  """
  train_data = pd.DataFrame()
  valid_data = pd.DataFrame()
  for i in range(len(dframe) - 1):
    idx1 = np.random.randint(len(dframe) - 1)
    idx2 = idx1 + 1
    
    row1 = dframe.iloc[[idx1]].reset_index()
    row2 = dframe.iloc[[idx2]].reset_index()
    
    randInt = np.random.randint(9)
    if 0 <= randInt <= 1:
      valid_frames = [valid_data, row1, row2]
      valid_data = pd.concat(valid_frames, axis = 0, join = 'outer', ignore_index=False)
    if randInt >= 2:
      train_frames = [train_data, row1, row2]
      train_data = pd.concat(train_frames, axis = 0, join = 'outer', ignore_index=False)
  return train_data, valid_data


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

def cv2_preprocess(row):
  now=cv2.imread(row['path_now'].values[0],0)
  now=now[100:350, 0:639]
  now=cv2.Canny(now,75,150)
  
  prev=cv2.imread(row['path_prev'].values[0],0)
  prev=prev[100:350, 0:639]
  prev=cv2.Canny(prev,75,150)
  
  flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  
  return flow.astype('float32')

def cv2_preprocess_tnsr_fcn(row):
  now=cv2.imread(row[1],0)
  now=now[100:350, 0:639]
  now=cv2.Canny(now,75,150)
  
  prev=cv2.imread(row[0],0)
  prev=prev[100:350, 0:639]
  prev=cv2.Canny(prev,75,150)
  
  flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
  
  return flow.astype('float32')

def image_tensor_func(df) :
  results = []
  for index, row in df.iterrows():
    flow_img = cv2_preprocess_tnsr_fcn(row)
    results.append( np.expand_dims( flow_img, axis=0 ) )
  return np.concatenate( results, axis = 0 )

def image_tensor_func_o(img4d) :
    results = []
    for img3d in img4d :
        rimg3d = image_func(img3d )
        results.append( np.expand_dims( rimg3d, axis=0 ) )
    return np.concatenate( results, axis = 0 )

class CustomLayer( Layer ) :
  def call( self, xin )  :
    xout = tf.py_func( image_tensor_func_o, 
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
  "1] read, save to npy, and plot the training speed data"
  if False:
    #read the train targets
    Y_train = np.genfromtxt('train.txt', delimiter=',')
    np.save('Y_train',Y_train)
    
    #plot the train targets
    time_train = np.arange(0,np.floor(20400/20),0.05)
    fig302, ax302 = plt.subplots()
    ax302.plot(time_train,Y_train, label='spd_train')
  else:
    Y_train = np.load('Y_train.npy')
  "end 1]"
  
  "2] convert the train video to images to be used for developing the preprocessinng pipeline"
  if False:
    # Read the video from specified path 
    cam = cv2.VideoCapture("C:\\git\\speedchallenge\\data\\train.mp4") 
    try: 
      # creating a folder named data 
      if not os.path.exists('data'): 
        os.makedirs('data') 
    # if not created then raise error 
    except OSError: 
      print ('Error: Creating directory of data') 
    # frame 
    currentframe = 0
      
    while(True): 
      # reading from frame 
      ret,frame = cam.read() 
      if ret: 
        # if video is still left continue creating images 
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
      else: 
        break
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 
  "end 2]"
  
  "4] create data frame of file paths to an image, the previous image, and the average speed from each image"
  if False:
    # create
    d = create_dframe()
    # save
    d.to_pickle("./df_img_paths_and_spd.pkl")
  else:
    # load
    d = pd.read_pickle("./df_img_paths_and_spd.pkl")
  
  "end 4]"
  
  "5] test cv2_preprocess and image_tensor_func"
  if False:
    dfrow1 = d.iloc[[1000]].reset_index()
    flow_test = cv2_preprocess(dfrow1)
    now=cv2.imread(dfrow1['path_now'].values[0],0)
    nowc = now[100:350, 0:639]
    cv2.imshow('flow1', draw_flow(nowc, flow_test))
    
    dfrow2 = d.iloc[[1001]].reset_index()
    flow_test2 = cv2_preprocess(dfrow2)
    now2=cv2.imread(dfrow2['path_now'].values[0],0)
    now2c = now[100:350, 0:639]
    cv2.imshow('flow2', draw_flow(now2c, flow_test2))
    
  if False:  
    dsml = d.iloc[[1000]]
    results = []
    for index, row in dsml.iterrows(): # this is a generator that produces data frame series, and not a single row dataframe
      flow_img = cv2_preprocess_tnsr_fcn(row)
      results.append( np.expand_dims( flow_img, axis=0 ) )
    #print(np.concatenate( results, axis = 0 ))
    
  if False:  
    dsml = d.iloc[1000:1002]
    results = []
    for index, row in dsml.iterrows(): # this is a generator that produces data frame series, and not a single row dataframe
      flow_img = cv2_preprocess_tnsr_fcn(row)
      print(flow_img.shape)
      results.append( np.expand_dims( flow_img, axis=0 ) )
    #print(np.concatenate( results, axis = 0 ))
    print(results[0].shape)
    
  if True:
    dsml = d.iloc[1000:1002]
    tnsr_fcn_test = image_tensor_func(dsml)
    #cv2.imshow('flow11', draw_flow(nowc, tnsr_fcn_test[0]))
    #cv2.imshow('flow12', draw_flow(nowc, tnsr_fcn_test[1]))
  "end 5]"
  
  "6] create test network for testing custom layer"
  if True:
    a = np.random.randn(2,100,200,3)
    #Layers
    x = Input(shape=(None,None,3))
    y = CustomLayer(name='custom')(x)
    #Models
    model = Model( inputs=x, outputs=y )
    #print model.summary()
    #test
    df_test = d.head(2)
    b = model.predict(df_test)
    
    
#class CustomLayer( Layer ) :
#  def call( self, xin )  :
#    xout = tf.py_func( image_tensor_func, 
#                       [xin],
#                       'float32',
#                       stateful=False,
#                       name='cvOpt')
#    xout = K.stop_gradient( xout ) # explicitly set no grad
#    xout.set_shape( [xin.shape[0], 250, 639, xin.shape[-1]] ) # explicitly set output shape
#    return xout
#  def compute_output_shape( self, sin ) :
#    return ( sin[0], 250, 639, sin[-1] )
  