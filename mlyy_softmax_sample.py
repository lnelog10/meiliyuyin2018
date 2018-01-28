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

VOICEDATA_LENTH = 13*25;#28*28=784
CLASSIFIER_TYPE = 17;#10
Pre_Train_Dir_Path = "./data_set/train"
Post_Train_Dir_Path = "./data_set/train_real"

# 语音输入
def voice_input():
  return



#读取录音sample里 mp3
def voice_read():
  return




# 1.语音输入
# 2.读取录音sample里 mp3

# 3.mp3输出对应嘴形视频
  # 3.1mp3 mfcc提取，生成txt
  # 3.2restore 网络参数，跑softmax 找到对应分类嘴形--->copy 到sample目录
  # 3.3sample里 mp3和嘴形合成视频
def main(_):
def main(_):

  return





if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
