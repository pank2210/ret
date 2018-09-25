
import os

import numpy as np
import pandas as pd
import sys

import tensorflow as tf

sys.path.append('../')

from util import config as cutil
from util import json_util as jutil
from util import myImg2 as myimg

class Data(object):
  def log(self, mname, msg, level=0):
    sep = '| '
    if level <= self.verbose:
      print("##" + "Data::" + mname + sep + msg)
   
  def __init__(self, id="rn1_",config=None,jfilepath="../../config/config.json"):
    mname = "__init__"
    
    self.id = id #shape of largest image unles restricted.
    self.img_buf_size = None #shape of largest image unles restricted.
     
    #Load / Initialize config source 
    if config == None:
      self.config = jutil.JsonUtil(jfilepath)
    else:
      self.config = config
   
    self.verbose = int(self.config.getElementValue(elem_path='/common/verbose'))
    #print("verbose[{}]".format(self.verbose)) 
    self.log( mname, "Initialized verbose[{}]".format(self.verbose), level=3)
    self.initialize_from_config()
   
  def initialize_from_config(self):
    mname = "initialize_from_config"
     
    self.img_buf_size = None #shape of largest image unles restricted.
    self.cdir = self.config.getElementValue(elem_path='/common/cdir')
    self.train_data_dir = self.config.getElementValue(elem_path='/common/data_dir_path')
    self.train_label_data_file = self.train_data_dir + self.config.getElementValue(elem_path='/train/label_data_file')
    self.log( mname, "Reading train_label_data_file[{}]".format(self.train_label_data_file), level=3)
    self.training_dataset_ratio = self.config.getElementValue(elem_path='/model/param/training_dataset_ratio')
    self.log( mname, "training_dataset_ratio[{}]".format(self.training_dataset_ratio), level=3)
    self.validation_dataset_ratio = self.config.getElementValue(elem_path='/model/param/validation_dataset_ratio')
    self.log( mname, "validation_dataset_ratio[{}]".format(self.validation_dataset_ratio), level=3)
    self.random_seed = self.config.getElementValue(elem_path='/model/param/random_seed')
    self.log( mname, "random_seed[{}]".format(self.random_seed), level=3)
     
    self.img_dir_path = self.config.getElementValue(elem_path='/img/img_dir_path')
    self.img_croped_dir_path = self.config.getElementValue(elem_path='/img/img_croped_dir_path')
    self.img_filename_ext = self.config.getElementValue(elem_path='/img/img_filename_ext')
    self.img_width = self.config.getElementValue(elem_path='/img/img_width')
    self.img_heigth = self.config.getElementValue(elem_path='/img/img_heigth')
     
    self.log( mname, "Images will be read from [{}]".format(self.img_dir_path), level=3)
    self.log( mname, "Image file extension [{}]".format(self.img_filename_ext), level=3)
    self.log( mname, "Image width [{}] heigth [{}]".format(self.img_width,self.img_heigth), level=3)
     
    self.img_processing_capacity = self.config.getElementValue(elem_path='/img/img_processing_capacity')
    self.log( mname, "img_processing_capacity[{}]".format(self.img_processing_capacity), level=3)
     
    self.channels = self.config.getElementValue(elem_path='/img/channels')
    self.log( mname, "channels[{}]".format(self.channels), level=3)
   
  def load_train_data(self):
    mname = "load_train_data"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_train = np.zeros(( 0, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h, 3), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
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
       
      #skip already processed data 
      if os.path.exists(self.img_croped_dir_path + rec.image + self.img_filename_ext):
        cnt += 1
        continue
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=imgpath) 
         
        i_w, i_h = myimg1.getImageDim() 
        croped_img_arr = np.zeros((n_img_w,n_img_h,3),dtype='uint8') 
        calc_img_w_offset = int((n_img_w - i_w)/2)
        calc_img_h_offset = int((n_img_h - i_h)/2)
        croped_img_arr[ calc_img_w_offset:(calc_img_w_offset + i_w), calc_img_h_offset:(calc_img_h_offset + i_h), :] = myimg1.getImage()
         
        ''' 
        croped_img = tf.image.resize_image_with_crop_or_pad( myimg1.getImage(), n_img_w, n_img_h)
        init = tf.global_variables_initializer()
        croped_img_arr = 0
        with tf.Session() as sess:
          sess.run(init)
          croped_img_arr = sess.run(croped_img)
          print(v.shape,type(v))  # will show you your variable.
          v = np.reshape( v, ( n_img_w, n_img_h, 3))
          print(v.shape,type(v))  # will show you your variable.
        ''' 
         
        x_img_buf[ 0, :, :, :] = croped_img_arr
         
        #'''#use below block of code to debug croped image with original.
        #myimg1.showImage()
        #myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=False)
        myimg2 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=rec.image+self.img_filename_ext, img=croped_img_arr) 
        myimg2.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
        #myimg2.saveImage()
        #'''
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
         
        #x_train = np.vstack( (x_train, x_img_buf))
        #x_train[cnt,:,:,:] = croped_img_arr
        y_buf.append(rec.level)
         
        self.df.loc[i,'imgexists'] = True
        self.df.loc[i,'w'], self.df.loc[i,'h'] = myimg1.getImageDim()
        self.df.loc[i,'_w'], self.df.loc[i,'_h'] = croped_img_arr.shape[0],croped_img_arr.shape[1]
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
     
   
  def load_data_as_greyscale(self):
    mname = "load_data_as_greyscale"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
    self.log( mname, "Loaded [{}] recs".format(self.df['level'].count()), level=3)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_train = np.zeros(( 0, n_img_w, n_img_h), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    for i,rec in self.df.iterrows():
      #if cnt >= 50:
      #  break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write("%6d out of %6d" % (cnt,tot_cnt))
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      #skip already processed data 
      if os.path.exists(self.img_croped_dir_path + rec.image + self.img_filename_ext):
        cnt += 1
        continue
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=imgpath) 
         
        myimg1.getGreyScaleImage(convertFlag=True) 
        myimg1.padImage(n_img_w,n_img_h)
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
         
        myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
         
        #x_train = np.vstack( (x_train, x_img_buf))
        #x_train[cnt,:,:,:] = croped_img_arr
        y_buf.append(rec.level)
         
        self.df.loc[i,'imgexists'] = True
        self.df.loc[i,'w'], self.df.loc[i,'h'] = myimg1.getImageDim()
        #self.df.loc[i,'_w'], self.df.loc[i,'_h'] = croped_img_arr.shape[0],croped_img_arr.shape[1]
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
   
  def load_img_data(self):
    mname = "load_greyscale_data"
     
    self.log( mname, "Loading Dataframe from [{}]".format(self.train_label_data_file), level=3)
    self.df = pd.read_csv( self.train_label_data_file)
    self.log( mname, "Loaded [{}] recs".format(self.df['level'].count()), level=3)
     
    #create & set all myImg Config 
    self.myImg_config = cutil.Config(configid="myConfId",cdir=self.cdir)
    self.myImg_config.setDdir( self.train_data_dir)
    self.myImg_config.setOdir( self.img_croped_dir_path)
    self.myImg_config.setIdir( self.img_dir_path)
     
    self.df['h'] = 0
    self.df['w'] = 0
    self.df['imgpath'] = ""
    self.df['imgexists'] = False
     
    #initialize all variables... 
    n_img_w = self.img_width
    n_img_h = self.img_heigth
     
    tot_cnt = self.img_processing_capacity 
    if tot_cnt == 0:
      tot_cnt = self.df['level'].count()
    cnt = 0
    file_missing = 0
     
    #generate dataset for handling train : test
    np.random.seed(self.random_seed)
    train_dataset_sample = np.random.choice( range(0,tot_cnt), int(tot_cnt * self.training_dataset_ratio), replace=False) 
    train_dataset_indicies = dict(zip(train_dataset_sample,train_dataset_sample))
     
    #x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h), dtype='uint8')
    x_train = None
    y_train = None
    x_test = None
    y_test = None
    y_train_buf = []
    y_test_buf = []
     
    if self.channels == 1:
      x_train = np.zeros(( len(train_dataset_sample), n_img_w, n_img_h), dtype='uint8')
      x_test = np.zeros(( (tot_cnt-len(train_dataset_sample)), n_img_w, n_img_h), dtype='uint8')
    else:
      x_train = np.zeros(( len(train_dataset_sample), n_img_w, n_img_h, self.channels), dtype='uint8')
      x_test = np.zeros(( (tot_cnt-len(train_dataset_sample)), n_img_w, n_img_h, self.channels), dtype='uint8')
    y_train = np.zeros((0,1),dtype='uint8')
     
    y_test = np.zeros((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    train_cnt = 0
    test_cnt = 0
    train_samples_cnt = len(train_dataset_sample)
    test_samples_cnt = tot_cnt - len(train_dataset_sample)
    self.log( mname, "[{}] recs for training.".format(train_samples_cnt), level=3)
    self.log( mname, "[{}] recs for test.".format(test_samples_cnt), level=3)
     
    for i,rec in self.df.iterrows():
      if cnt >= tot_cnt:
        break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write(progress_sts)
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
      if os.path.exists(imgpath):
        myimg1 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=imgpath) 
         
        #x_img_buf[ 0, :, :] = myimg1.getImage()
        if train_dataset_indicies.get(cnt,False): 
          #x_train = np.vstack( (x_train, x_img_buf))
          if train_cnt < train_samples_cnt:
            if self.channels == 1:
              x_train[train_cnt,:,:] = myimg1.getImage()
            else:
              x_train[train_cnt,:,:,:] = myimg1.getImage()
          y_train_buf.append(rec.level)
          train_cnt += 1
        else:
          #x_test = np.vstack( (x_test, x_img_buf))
          #self.log( mname, "[{}] [{}] x_test[{}] x_img_buf[{}]".format(cnt,test_cnt,x_test.shape,x_img_buf.shape), level=2)
          if test_cnt < test_samples_cnt:
            if self.channels == 1:
              x_test[test_cnt,:,:] = myimg1.getImage()
            else:
              x_test[test_cnt,:,:,:] = myimg1.getImage()
          y_test_buf.append(rec.level)
          test_cnt += 1
         
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
      else:
        #self.log( mname, "Image file [{}] doesn't exists!!!".format(imgpath), level=2)
        file_missing += 1
       
      cnt += 1
      
    #create y array as required
    y_train = np.array( y_train_buf, dtype='uint8')
    y_train = np.reshape( y_train, (y_train.size,1))
    y_test = np.array( y_test_buf, dtype='uint8')
    y_test = np.reshape( y_test, (y_test.size,1))
    #print final dimensionf or x_train and y_train
    self.log( mname, "x_train [{}] y_train [{}]".format(x_train.shape,y_train.shape), level=3)
    self.log( mname, "x_test [{}] y_test [{}]".format(x_test.shape,y_test.shape), level=3)
      
    self.log( mname, "Process dataset [{}]".format(cnt), level=3)
    self.log( mname, "File missing [{}]".format(file_missing), level=3)
    self.log( mname, "Max image width[{}] heigth[{}]".format(self.df['w'].max(),self.df['h'].max()), level=3)
    #print(self.df.head(10))
    #self.df.to_csv( self.train_data_dir + 'u_img_set.csv')
    
    return (x_train, y_train), (x_test, y_test)
     
     
if __name__ == "__main__":
  #prep_data()
  data = Data()
  #data.load_train_data()
  #data.load_data_as_greyscale()
  data.load_img_data()
