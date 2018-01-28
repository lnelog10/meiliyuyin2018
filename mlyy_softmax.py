# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
import mlyy_dataset as input_data
import os;
import glob as gb
import os;
import scipy.misc
import numpy
import numpy as np
from mlyy_softmax_sample_utils import ffmpegGenVideo

VOICEDATA_LENTH = 13*25;#28*28=784
CLASSIFIER_TYPE = 6;#10
Pre_Train_Dir_Path = "./data_set/train"
Post_Train_Dir_Path = "./data_set/train_real"

def main(_):
  # Import data
  mlyy_dataset = input_data.read_data_sets(Post_Train_Dir_Path, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, VOICEDATA_LENTH])
  W = tf.Variable(tf.zeros([VOICEDATA_LENTH, CLASSIFIER_TYPE]))
  b = tf.Variable(tf.zeros([CLASSIFIER_TYPE]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, CLASSIFIER_TYPE])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  if load():
    print(" [*] Load SUCCESS")
  else:
    print(" [!] Load failed...")
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mlyy_dataset.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  save();

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mlyy_dataset.test.images,
                                      y_: mlyy_dataset.test.labels}))

  print("sample_model...")
  sample_model(W=W,b=b)

gen_sample_voices = "./sample/gen_sample_voices/"  # *.txt
gen_sample_images = "./sample/gen_sample_images/"  # *.jpg
sample_mp3 = "./sample/K.mp3"
out_mp4 = "./sample/K.mp4"
# 3.mp3输出对应嘴形视频
# 3.1mp3 mfcc提取，生成txt
# 3.2restore 网络参数，跑softmax 找到对应分类嘴形--->copy 到sample目录
# 3.3sample里 mp3和嘴形合成视频
def sample_model(self,W,b,sampleVoiceTxts=gen_sample_voices):
    print('extract_voices', sampleVoiceTxts)
    rootExt = sampleVoiceTxts + os.sep + "*"
    txts = gb.glob(rootExt)
    voiceFeatures = np.zeros((txts.__len__(), 13, 25))
    for index in range(len(txts)):
      voiceFeatures[index] = np.loadtxt(txts[index]);
      y = tf.matmul(voiceFeatures[index], W) + b
      #取出序号 TODO
      cat =y
      #保存图片
      scipy.misc.imsave(gen_sample_images, getSampleImgName(voicePath=txts[index],sample_dir=gen_sample_voices,))
    print("ffmpegGenVideo...")
    ffmpegGenVideo(imageSlicesDir=gen_sample_images,mp3SampleFile=sample_mp3,outfile=out_mp4)

def getSampleImgName(voicePath,sample_dir):
    fileName = voicePath[voicePath.rindex("/")+1:]
    temp = fileName.split('.')
    result = '{}{}.jpg'.format(sample_dir,temp[0])
    return result


def save(self, checkpoint_dir="checkpoint_dir",model_dir="model_dir"):
  model_name = "pix2pix.model"
  # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name))


def load(self, checkpoint_dir="checkpoint_dir",model_dir="model_dir"):
  print(" [*] Reading checkpoint...")
  # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
  checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and ckpt.model_checkpoint_path:
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    return True
  else:
    return False

if __name__ == '__main__':
  with tf.Session() as sess:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
