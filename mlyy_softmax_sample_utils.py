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
from pydub import AudioSegment
import librosa
import numpy as np
import argparse

VOICE_FRAGMENT_LENGTH = 250

def ffmpegGenVideo(imageSlicesDir,mp3SampleFile,outfile):
    # os.system("ffmpeg -threads2 -y -r 4 -i "+imageSlicesDir+"image%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc "+outfile)
    # -r 是frame rate
    os.system("ffmpeg -y -r 3 -i "+imageSlicesDir+"K%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc -strict -2 "+outfile)

def sample_voice_process(dir, SampleVoice):
    song = AudioSegment.from_mp3(SampleVoice)
    sum = int(song.__len__()/VOICE_FRAGMENT_LENGTH)
    # sum = sum+1 最后不能整除的丢弃
    for i in range(sum):
        next = (i + 1) * VOICE_FRAGMENT_LENGTH
        first_10_seconds = song[i * VOICE_FRAGMENT_LENGTH:next]
        index =(i + 1);
        mp3name = dir + '{:04d}.mp3'.format(index)
        first_10_seconds.export( mp3name, format="mp3")
        y1, sr1 = librosa.load(mp3name, sr=16000)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=190, n_fft=2048)#13*25
        print("==>"+str(i)+"<==",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(dir+'{:04d}.txt'.format(index),mfccs)

def ffmpegTrans2mp3(wmvPath,mp3Path):
    FFMPEG_EXEC_COMMAND = \
      "ffmpeg -i %s -f mp3 -vn %s" % (wmvPath, mp3Path)
    os.system(FFMPEG_EXEC_COMMAND)