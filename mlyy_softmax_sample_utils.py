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
import os
specifiedSmapleImagePath = "./sample/K.jpg"
sample_mp3 = "./sample/K.mp3"
out_mp4 = "./sample/K.mp4"

# 3.mp3输出对应嘴形视频
  # 3.1mp3 mfcc提取，生成txt
  # 3.2restore 网络参数，跑softmax 找到对应分类嘴形--->copy 到sample目录
  # 3.3sample里 mp3和嘴形合成视频
def mp3ToMfccs(mp3Path):
  gen_sample_voices = "./sample/gen_sample_voices/"  # *.txt
  gen_sample_images = "./sample/gen_sample_images/"  # *.jpg
  # gen_sample_images_his = "./datasets/first_run/sample/gen_sample_images_his/"  # *.jpg 历史数据
  # voice
  # sample_voice_process(genSampleVoice=gen_sample_voices,SampleVoice=sample_mp3)
  data = glob(gen_sample_voices + '*.txt')
  for voicepath in data:
    print("deal with voice", voicepath)
    voiceData = np.loadtxt(voicepath)
    print("voice shape", voiceData.shape)
    voiceData = voiceData.reshape(1, 13, 35, 1)
    voiceData = np.array(voiceData).astype(np.float32)
    samples = self.sess.run([self.fake_image_sample], feed_dict={self.sample_random_image: specifiedSmapleImage,
                                                                 self.sample_real_voice: voiceData})
    # print(tf.shape(samples))
    samples = np.reshape(samples, [112, 112, 3])
    so_save_image(samples, getSampleImgName(voicepath, gen_sample_images))
    # so_save_image(samples, getSampleImgNameHis(voicepath, gen_sample_images_his, epoch, idx, ))#历史记录
    # print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
  ffmpegGenVideo(imageSlicesDir=gen_sample_images, mp3SampleFile=sample_mp3, outfile=out_mp4)
  return


def ffmpegGenVideo(imageSlicesDir,mp3SampleFile,outfile):
    # os.system("ffmpeg -threads2 -y -r 4 -i "+imageSlicesDir+"image%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc "+outfile)
    # -r 是frame rate
    os.system("ffmpeg -y -r 3 -i "+imageSlicesDir+"K%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc -strict -2 "+outfile)



