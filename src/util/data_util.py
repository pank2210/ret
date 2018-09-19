
import os

import numpy as np
import pandas as pd
import sys

import tensorflow as tf

import config as cutil
import json_util as jutil
import myImg2 as myimg

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
     
    self.img_dir_path = self.config.getElementValue(elem_path='/img/img_dir_path')
    self.img_croped_dir_path = self.config.getElementValue(elem_path='/img/img_croped_dir_path')
    self.img_filename_ext = self.config.getElementValue(elem_path='/img/img_filename_ext')
    self.img_width = self.config.getElementValue(elem_path='/img/img_width')
    self.img_heigth = self.config.getElementValue(elem_path='/img/img_heigth')
    self.log( mname, "Images will be read from [{}]".format(self.img_dir_path), level=3)
    self.log( mname, "Image file extension [{}]".format(self.img_filename_ext), level=3)
    self.log( mname, "Image width [{}] heigth [{}]".format(self.img_width,self.img_heigth), level=3)
   
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
     
    x_train = np.zeros(( tot_cnt, n_img_w, n_img_h, 3), dtype='uint8')
    x_img_buf = np.empty(( 1, n_img_w, n_img_h, 3), dtype='uint8')
    y_buf = []
    y_train = np.empty((0,1),dtype='uint8')
     
    #loop in through dataframe. 
    for i,rec in self.df.iterrows():
      if cnt > 50:
        break
       
      progress_sts = "%6d out of %6d" % (cnt,tot_cnt)
      sys.stdout.write("%6d out of %6d" % (cnt,tot_cnt))
      sys.stdout.write("\b" * len(progress_sts)) # return to start of line, after '['
      sys.stdout.flush()
       
      imgpath = self.img_dir_path + rec.image + self.img_filename_ext 
      self.df.loc[i,'imgpath'] = imgpath
       
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
         
        #x_img_buf[ 0, :, :, :] = croped_img_arr
         
        #'''#use below block of code to debug croped image with original.
        #myimg1.showImage()
        #myimg1.saveImage(img_type_ext='.jpeg',gen_new_filename=False)
        #myimg2 = myimg.myImg( imageid=str(i), config=self.myImg_config, path=rec.image+self.img_filename_ext, img=croped_img_arr) 
        #myimg2.saveImage(img_type_ext='.jpeg',gen_new_filename=True)
        #myimg2.saveImage()
        #'''
         
        #self.log( mname, "Croped Image [{}] [{}] [{}] [{}]".format(myimg1.getImage().shape,croped_img_arr.shape,x_train.shape,x_img_buf.shape), level=4)
         
        #x_train = np.vstack( (x_train, x_img_buf))
        x_train[cnt,:,:,:] = croped_img_arr
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
     
if __name__ == "__main__":
  #prep_data()
  data = Data()
  data.load_train_data()
