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
import glob as gb
import os;
import shutil
import numpy as np
from mlyy_softmax_sample_utils import ffmpegGenVideo
from mlyy_softmax_sample_utils import ffmpegTrans2mp3
from mlyy_softmax_sample_utils import sample_voice_process
from AudioUtils import recordAndSaveAudio
import AudioUtils
import random

VOICEDATA_LENTH = 13*25;#28*28=784
CLASSIFIER_TYPE = 17;#10
Pre_Train_Dir_Path = "./data_set/train"
Post_Train_Dir_Path = "./data_set/train_real"
gen_sample_voices = "./sample/gen_sample_voices/"  # *.txt
gen_sample_images = "./sample/gen_sample_images/"  # *.jpg
sample_mp3 = "./sample/K.mp3"
out_mp4 = "./sample/K.mp4"



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
  # self.sess = sess
  # self.saver = tf.train.Saver()

  tf.global_variables_initializer().run()
  # if load():
  #   print(" [*] Load SUCCESS")
  # else:
  #   print(" [!] Load failed...")
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mlyy_dataset.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  # save();

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mlyy_dataset.test.images,
                                      y_: mlyy_dataset.test.labels}))

  print("sample_model...")
  recordAndSaveAudio()
  # 转化mp3
  ffmpegTrans2mp3(wmvPath=AudioUtils.WAVE_OUTPUT_FILENAME,mp3Path=sample_mp3);
  # 音频mfcc信息的提取
  sample_voice_process(dir=gen_sample_voices, SampleVoice=sample_mp3)
  sample_model(W=W,b=b)


def loadTxt(filename):
    temp = np.loadtxt(filename).reshape(13*25)
    #print("temp:"+temp)
    return temp


# 3.mp3输出对应嘴形视频
# 3.1mp3 mfcc提取，生成txt
# 3.2restore 网络参数，跑softmax 找到对应分类嘴形--->copy 到sample目录
# 3.3sample里 mp3和嘴形合成视频
def sample_model(W,b,sampleVoiceTxts=gen_sample_voices):
    print('extract_voices', sampleVoiceTxts)
    rootExt = sampleVoiceTxts + "*.txt"
    txts = gb.glob(rootExt)
    voiceFeatures = np.zeros((txts.__len__(), 13, 25))
    sess = tf.InteractiveSession()
    txtsData = [loadTxt(batch_file) for batch_file in txts]
    txtsData = np.array(txtsData).astype(np.float32)
    sampleX = tf.placeholder(tf.float32, [None, 13*25])

    y = tf.matmul(sampleX, W) + b

    sess = tf.InteractiveSession()
    # self.sess = sess
    # self.saver = tf.train.Saver()

    tf.global_variables_initializer().run()

    print(sess.run(W))
    print(sess.run(b))
    # tf.Print(data=W)
    # tf.Print(data=b)
    # if load():
    #   print(" [*] Load SUCCESS")
    # else:
    #   print(" [!] Load failed...")
    # Train
    results = sess.run(y, feed_dict={sampleX: txtsData})
    #print("results:"+results)
    for i in range(len(results)):
        result = results[i]
        print("result:"+str(result))
        k = getIndexOfY(result)
        print("index:"+str(k))
        srcFile = "./{:04d}.jpg".format(k);
        shutil.copy(srcFile, getSampleImgName( i , sample_dir=gen_sample_images))

    print("ffmpegGenVideo...")
    ffmpegGenVideo(imageSlicesDir=gen_sample_images,mp3SampleFile=sample_mp3,outfile=out_mp4)

def getIndexOfY(y):
    # for k in range(len(y)):
        # if y[k] != 0:
            # return k+1
    return random.randint(1,17)

def getSampleImgName(time, sample_dir):
    # fileName = voicePath[voicePath.rindex("/")+1:]
    # temp = fileName.split('.')
    result = '{}/{:04d}.jpg'.format(sample_dir,time)
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
