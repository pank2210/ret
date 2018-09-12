
import os

import numpy as np

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from tensorflow.python.keras.utils.data_utils import get_file

import config as cutil
import json_util as jutil
import myImg as myimg

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
  prep_data()
