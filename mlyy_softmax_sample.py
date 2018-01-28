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




# 语音输入
# 读取录音sample里 mp3
# mp3输出对应嘴形视频
# mp3 mfcc提取，生成txt
# restore 网络参数，跑softmax 找到对应分类嘴形--->copy 到sample目录
# sample里 mp3和嘴形合成视频
def main(_):
  return




def save(self, checkpoint_dir, step):
  return
        # model_name = "pix2pix.model"
        # model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        # if not os.path.exists(checkpoint_dir):
        #     os.makedirs(checkpoint_dir)
        #
        # self.saver.save(self.sess,
        #                 os.path.join(checkpoint_dir, model_name),
        #                 global_step=step)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
