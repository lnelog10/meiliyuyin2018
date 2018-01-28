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

def ffmpegGenVideo(imageSlicesDir,mp3SampleFile,outfile):
    # os.system("ffmpeg -threads2 -y -r 4 -i "+imageSlicesDir+"image%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc "+outfile)
    # -r 是frame rate
    os.system("ffmpeg -y -r 3 -i "+imageSlicesDir+"K%04d.jpg -i "+mp3SampleFile+" -absf aac_adtstoasc -strict -2 "+outfile)



