import cv2
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, Dense, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

def generator(d,batch_size=None):
  """
  generator: shuffle, read RGB images as greyscale, crop, detect edges, compute optical flow
  """
  while True:
    for i in np.arange(d.shape[0]):
      prev = cv2.imread(d['path_prev'].iloc[i],0)
      now = cv2.imread(d['path_now'].iloc[i],0)
      # crop to 250x640 to remove sky and dashboard
      prev=prev[100:350, :]
      now =now[100:350, :]
      # edge detection
      prev=cv2.Canny(prev,75,150)
      now=cv2.Canny(now,75,150)
      # compute optical flow
      flow = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
      yield ({'input_1': flow}, {'output': d['spd'].iloc[i]})
  
def generate_np_array(d,batch_size=None):
  """
  read RGB images as greyscale, crop, detect edges, compute optical flow
  """  
  if batch_size is None:
    batch_size = d.shape[0]
  img4d = np.zeros([batch_size,250,640,2])# comma speed challenge video frames are 480x640 pixels
  spd = np.zeros(batch_size)
  for row in np.arange(batch_size):
    prev = cv2.imread(d['path_prev'].iloc[row],0)
    now = cv2.imread(d['path_now'].iloc[row],0)
    # crop
    prev=prev[100:350, :]
    now=now[100:350, :]
    # edge detection
    prev=cv2.Canny(prev,75,150)
    now=cv2.Canny(now,75,150)
    # compute optical flow
    img4d[row,] = cv2.calcOpticalFlowFarneback(prev, now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    spd[row] = d['spd'].iloc[row]
  return img4d, spd

def batch_shuffle(d,validation_split=0.2):
  """
  Randomly split the data into training and validation sets
  """
  train_data = pd.DataFrame()
  valid_data = pd.DataFrame()
  for i in range(len(d) - 1):
    idx1 = np.random.randint(len(d) - 1)
    row = d.iloc[[idx1]].reset_index()
    
    randInt = np.random.randint(100)
    if 0 <= randInt <= 20:
      valid_frames = [valid_data, row]
      valid_data = pd.concat(valid_frames, axis = 0, join = 'outer', ignore_index=False)
    if randInt >= 21:
      train_frames = [train_data, row]
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
  
  "3] create data frame of file paths to an image, the previous image, and the average speed from each image"
  if False:
    # create
    d = create_dframe()
    # save
    d.to_pickle("./df_img_paths_and_spd.pkl")
  else:
    # load
    d = pd.read_pickle("./df_img_paths_and_spd.pkl")
  
  "end 3]"
  
  "4] test generator"
  if False:
    img4d,spd = generator(d,batch_size=1,gen_test_flg=True)

    now=cv2.imread(d['path_now'].iloc[0],0)
    nowc = now[100:350, :]
    cv2.imshow('CL1', draw_flow(nowc, img4d[0,]))
  "end 4]"
  
  "5] suffle train data to create trainnig and validation sets"
  if False:
    train_data, valid_data = batch_shuffle(d)
  "end 5]"
  
  "5.5] dont use generator, but just create a large np array of inputs, and corresponding np array of targets"
  if True:
    x, y = generate_np_array(train_data,batch_size=None)
    xv, yv = generate_np_array(valid_data,batch_size=None)
    dont_use_generator = True
  else:
    dont_use_generator = False
  "end 5.5]"
  
  "6] create baseline net"
  if True:
    model = Sequential()
    model.add(Dense(1000, input_shape=(250,640,2), activation='relu', kernel_initializer='normal'))
#    model.add(Conv2D(24, 5, 
#                     strides=(5,5),
#                     input_shape=(250,640,2),
#                     data_format='channels_last',
#                     activation='relu'))
#    model.add(Conv2D(36, 5,
#                     strides=(5,5),
#                     data_format='channels_last',
#                     activation='relu'))
#    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='normal'))
    #model.add(Dropout(n_dropout))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
  "end 6]"
  
  "7] fit model"
  if True:
    clbkList = [EarlyStopping(monitor='loss', min_delta=0.4, patience=1, verbose=1)]
    if dont_use_generator:
      model.fit(x=x, y=y, epochs=10, verbose=2,callbacks=clbkList, shuffle=False, validation_data=(xv, yv))
    else:
      model.fit_generator(generator(train_data), steps_per_epoch=100, epochs=2, verbose=1,callbacks=clbkList)
   
  "8] evaluate model"
  if False:
    #evaluate_generator(generator, steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=0)
    model.evaluate_generator(generator(valid_data), steps=None, callbacks=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)