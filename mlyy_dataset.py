# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Functions for  reading MeiLiYuYin data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob as gb
import os;

import numpy
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
import mlyy_softmax as main

#data size:15841

def read_data_sets(train_dir,
                   one_hot=True,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=1500,
                   seed=None):
  train_images,train_labels = extract_voices(train_dir)
  train_images = np.array(train_images).astype(np.float32)
  train_labels = np.array(train_labels).astype(np.float32)

  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  validation_images = train_images[750:validation_size]
  validation_labels = train_labels[750:validation_size]

  test_images = train_images[0:750]
  test_labels = train_labels[0:750]

  options = dict(dtype=dtype, reshape=reshape, seed=seed)
  train = DataSet(train_images, train_labels, **options)
  validation = DataSet(validation_images, validation_labels, **options)
  test = DataSet(test_images, test_labels, **options)
  return base.Datasets(train=train,validation=validation,test=test)

def extract_voices(root):
  print('extract_voices', root)
  rootExt= root+os.sep+"*"
  txts = gb.glob(rootExt)
  print('all data size:'+str(txts.__len__()))
  voiceFeatures=np.zeros((txts.__len__(),13,25))
  lables=np.zeros(txts.__len__(),dtype=np.int)
  for index in range(len(txts)):
    baseName = os.path.basename(txts[index]);
    if not (baseName.startswith("007")
            or baseName.startswith("008")
            ):
      print(txts[index])
      voiceFeatures[index] = np.loadtxt(txts[index]);
      # print('voiceFeature'+i, voiceFeatures[i])
      #
      basename = os.path.basename(txts[index])
      filename, extension = os.path.splitext(basename)
      args = filename.split("_")
      lables[index] = args[1]
      # print('lables' + i,lables[i])

  result_lables = dense_to_one_hot(lables, num_classes=main.CLASSIFIER_TYPE)
  return voiceFeatures,result_lables

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot



class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 1000 #TODO
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        # assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        # images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]



def load_mnist(train_dir='./data_set/train_real'):
  return read_data_sets(train_dir)
