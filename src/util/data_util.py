
import os

import numpy as np
import pandas as pd
import sys

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

import tensorflow as tf

import config as cutil
import json_util as jutil
import myImg as myimg

class Data(object):
  def log(self, mname, msg, level=0):
    sep = '| '
    if level <= self.verbose:
      print("##" + "Data::" + mname + sep + msg)
   
  def __init__(self, id="rn1_",config=None):
    mname = "__init__"
    
    self.id = id #shape of largest image unles restricted.
    self.img_buf_size = None #shape of largest image unles restricted.
     
    #Load / Initialize config source 
    if config == None:
      self.config = jutil.JsonUtil()
    else:
      self.config = config
   
    self.verbose = int(self.config.getElementValue(elem_path='/common/verbose'))
    #print("verbose[{}]".format(self.verbose)) 
    self.log( mname, "Initialized verbose[{}]".format(self.verbose), level=3)
    self.initialize_from_config()
   
  def initialize_from_config(self):
    mname = "load_train_data"
     
    self.img_buf_size = None #shape of largest image unles restricted.
    self.cdir = self.config.getElementValue(elem_path='/common/cdir')
    self.train_data_dir = self.config.getElementValue(elem_path='/common/data_dir_path')
    self.train_label_data_file = self.train_data_dir + self.config.getElementValue(elem_path='/train/label_data_file')
    self.log( mname, "Reading train_label_data_file[{}]".format(self.train_label_data_file), level=3)
     
    self.img_dir_path = self.config.getElementValue(elem_path='/common/img_dir_path')
    self.img_filename_ext = self.config.getElementValue(elem_path='/common/img_filename_ext')
    self.log( mname, "Images will be read from [{}]".format(self.img_dir_path), level=3)
    self.log( mname, "Image file extension [{}]".format(self.img_filename_ext), level=3)
   
  def load_train_data(self):
    mname = "load_train_data"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.train_data_dir)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = 3000
    n_img_h = 5000
     
    x_train = np.empty(( 0, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h, 3), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
    tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #loop in through dataframe. 
    for i,rec in self.df.iterrows():
      #if cnt > 50:
      #  break
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write("%6d out of %6d" % (cnt,tot_cnt))
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()

       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=imgpath) 
        croped_img = tf.image.resize_image_with_crop_or_pad( myimg1.getImage(), n_img_w, n_img_h)
         
        init = tf.global_variables_initializer()
        croped_img_arr = 0
        with tf.Session() as sess:
          sess.run(init)
          croped_img_arr = sess.run(croped_img)
          '''
          print(v.shape,type(v))  # will show you your variable.
          v = np.reshape( v, ( n_img_w, n_img_h, 3))
          print(v.shape,type(v))  # will show you your variable.
          '''
         
        x_img_buf[ 0, :, :, :] = croped_img_arr
         
        '''#use below block of code to debug croped image with original.
        #myimg1.showImage()
        myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=False)
        myimg2 = myimg.myImg( imageid='X1', config=self.myImg_config, path=None, img=croped_img_arr) 
        myimg2.saveImage(img_type_ext='.jpeg',gen_new_filename=False)
        '''
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
        x_train = np.vstack( (x_train, x_img_buf))
        y_buf.append(rec.level)
         
        self.df.loc[i,'imgexists'] = True
        self.df.loc[i,'w'], self.df.loc[i,'h'] = myimg1.getImageDim()
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      cnt += 1
      
    #create y array as required
    y_train = np.array( y_buf, dtype='uint8')
    y_train = np.reshape( y_train, (y_train.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_train [{}] y_train [{}]".format(x_train.shape,y_train.shape), level=3)
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    self.df.to_csv( self.train_data_dir + 'u_img_set.csv')
     
def prep_data():
  """Loads CIFAR10 dataset.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  print("##data_prep called...")
  i_cdir = "../../"
  i_imgpath = "1000_left.jpeg"
  config = cutil.Config(configid="myConfId",cdir=i_cdir)
  img1 = myimg.myImg(imageid="xx",config=config,ekey='x123',path=i_imgpath)
  img1.printImageProp()
  
  train_samples = 1 
  w, h = img1.getImageDim()
  #x_train1 = np.empty(( train_samples, 3, w, h), dtype='uint8')
  x_train1 = np.empty(( train_samples, w, h, 3), dtype='uint8')
  print(" x_train1 size [{}]".format(x_train1.shape))
  x_train1[ 0, :, :, :] = img1.getImage()
  print(" x_train1 size [{}]".format(x_train1.shape))
  #x_train1 = x_train1.transpose(0, 2, 3, 1)
  #print(" x_train1 size [{}]".format(x_train1.shape))  
  
   
  dirname = 'cifar-10-batches-py'
  #./.keras/datasets/cifar-10-batches-py.tar.gz
  origin = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  origin = 'file://Users/pankaj.petkar/.keras/datasets/cifar-10-batches-py.tar.gz'
  path = get_file(dirname, origin=origin, untar=True)

  num_train_samples = 50000

  x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
  y_train = np.empty((num_train_samples,), dtype='uint8')

  for i in range(1, 6):
    fpath = os.path.join(path, 'data_batch_' + str(i))
    (x_train[(i - 1) * 10000:i * 10000, :, :, :],
     y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

  print(x_train.shape)
  print(y_train.shape)

  fpath = os.path.join(path, 'test_batch')
  x_test, y_test = load_batch(fpath)

  y_train = np.reshape(y_train, (len(y_train), 1))
  y_test = np.reshape(y_test, (len(y_test), 1))

  if K.image_data_format() == 'channels_last':
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)
   
  print(x_train.shape)
  print(y_train.shape)

  return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
  #prep_data()
  data = Data()
  data.load_train_data()
