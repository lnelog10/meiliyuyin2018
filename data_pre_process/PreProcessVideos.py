import glob
import os
import shutil
import numpy as np
import librosa
from pydub import AudioSegment

import dlib
from imutils import face_utils
import cv2
import math

"""
处理将mp3文件提取成小片段，然后提取mfcc信息，以prefix + 四位编号.txt的格式，保存在outputDir里面
"""

VOICE_FRAGMENT_LENGTH = 250
def sample_voice_process(prefix, SampleVoice, outputDir):
    song = AudioSegment.from_mp3(SampleVoice)
    sum = int(song.__len__()/VOICE_FRAGMENT_LENGTH)
    for i in range(sum):
        next = (i + 1) * VOICE_FRAGMENT_LENGTH
        first_10_seconds = song[i * VOICE_FRAGMENT_LENGTH:next]
        index =(i + 1);
        mp3name = os.path.join(outputDir,prefix + '{:04d}.mp3'.format(index))
        # 一般就不生成mp3了
        first_10_seconds.export( mp3name, format="mp3")
        y1, sr1 = librosa.load(mp3name, sr=16000)
        mfccs = librosa.feature.mfcc(y=y1, sr=sr1, n_mfcc=13, hop_length=190, n_fft=2048)#13*35
        print("==>"+str(i)+"<==",len(mfccs).__str__() + "*" + len(mfccs[0]).__str__())
        np.savetxt(os.path.join(outputDir,prefix+'{:04d}.txt'.format(index)),mfccs)


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")
"""
处理将mp4文件的每帧取出，然后截取人脸区域（大小一般为四倍人脸大小），以prefix + 四位编号.jpg的格式，保存在outputDir里面
"""
def sampe_image_process(prefix, sampleVideo, outputDir):
    cap = cv2.VideoCapture(sampleVideo)

    print("frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("frame rate:", cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    (x, y, w, h) = (None, None, None, None)

    time_count = -1
    SLOT_TIME = 250
    time_stamp = 0

    while (cap.isOpened()):
        time_count = time_count + 1
        # 250ms中取中间那一帧
        time_stamp = time_count * SLOT_TIME + 125
        cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)
        ret, image = cap.read()
        if ret == True:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            print("====================================================>")
            print("big image type:", image.dtype)
            print("gray image type:", gray.dtype)

            # 取第一张人脸
            if len(rects) > 0:
                rect = rects[0]
            else:
                continue
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            (x1, y1, w1, h1) = (x, y, w, h)

            (desireHeight, desireWidth) = image.shape[:2]
            # desireHeight = desireWidth

            # 取眼睛最左最右的坐标
            (leftEyeOutX, leftEyeOutY) = shape[36]
            (rightEyeOutX, rightEyeOutY) = shape[45]
            (leftEyeInnerX, leftEyeInnerY) = shape[39]
            (rightEyeInnerX, rightEyeInnerY) = shape[42]

            leftEyeY = (leftEyeOutY + leftEyeInnerY) / 2
            leftEyeX = (leftEyeOutX + leftEyeInnerX) / 2
            rightEyeY = (rightEyeOutY + rightEyeInnerY) / 2
            rightEyeX = (rightEyeOutX + rightEyeInnerX) / 2

            # 取左右眼的中心点坐标
            eyesCenterY = ((int)((leftEyeY + rightEyeY) / 2))
            eyesCenterX = ((int)((leftEyeX + rightEyeX) / 2))

            # 计算旋转角度（以左右眼中心点为十字坐标）
            dy = (rightEyeY - leftEyeY)
            dx = (rightEyeX - leftEyeX)
            length = cv2.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx) * 180.0 / np.pi
            scale = 1

            rows, cols = ((y1 + h1), (x1 + w1))
            M = cv2.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, 1)
            image = cv2.warpAffine(image, M, (desireWidth, desireHeight))

            # 再检测下是否能取到人脸，能取到才保存下来
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # 只取第一张人脸,
            if len(rects) > 0:
                rect = rects[0]
            else:
                continue

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            # 取脸上下左右的坐标
            (leftX, leftY) = shape[0]
            (rightX, rightY) = shape[16]
            (topX, topY) = shape[27]
            (bottomX, bottomY) = shape[57]
            finalWidth = (rightX - leftX)
            finalHeight = (bottomY - topY)

            # 这里要稍微截下人脸区域，防止图片过大，这里取四倍人脸
            centerFaceX_left = leftX - 2 * finalWidth
            if centerFaceX_left < 0:  #超出原图的区域
                centerFaceX_left = 0

            centerFaceX_right = rightX + 2 * finalWidth
            if centerFaceX_right > image.shape[1]:
                centerFaceX_right = image.shape[1]

            centerFaceY_top = topY - finalHeight * 2
            if centerFaceY_top < 0:
                centerFaceY_top = 0

            centerFaceY_bottom = bottomY + finalHeight * 2
            if centerFaceY_bottom > image.shape[0]:
                centerFaceY_bottom = image.shape[0]

            image = image[centerFaceY_top:centerFaceY_bottom,
                    centerFaceX_left:centerFaceX_right]

            # 可能的错误，类似判空,防止异常退出
            try:
                #						image = imutils.resize(image, width=112)
                if image.shape[0] == 0 or image.shape[1] == 0:
                    image = None
                else:
                    pass
                    # image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
            except ZeroDivisionError:
                image = None

            if image is None:
                pass
            else:
                print("time count:", time_count)
                print("final image shape :", image.shape)

                #格式是，保存到outputDir指定的目录下，前后加上前缀，再加当前帧的编号
                outputFileName = os.path.join(outputDir, prefix + "{:04d}.jpg".format(time_count + 1))
                cv2.imwrite(outputFileName, image)

            # 可有可无，这里不接收键盘输入的
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:  # 最后一帧了，直接跳出来
            break;

    cap.release()
    cv2.destroyAllWindows()

def processPerVideo(basename):
    print("[info] now process "+basename)
    #以mp4的前缀名创建一个目录
    if not os.path.exists(basename):
        os.mkdir(basename)

    MP4_NAME = basename+".mp4"
    MP3_NAME = os.path.join(basename,basename+".mp3")
    #提取mp3并放入对应的目录下
    print("[info] fetch %s from %s" % (MP4_NAME, MP3_NAME))
    FFMPEG_EXEC_COMMAND = \
        "ffmpeg -i %s -f mp3 -vn %s" % (MP4_NAME, MP3_NAME)
    os.system(FFMPEG_EXEC_COMMAND)
    #音频mfcc信息的提取
    sample_voice_process(basename,MP3_NAME,basename)

    #把mp4复制到这个目录下
    shutil.copy(MP4_NAME, basename)
    MP4_NAME = os.path.join(basename, MP4_NAME)
    sampe_image_process(basename, MP4_NAME, basename)

def main():
    mp4_filenames = glob.glob("*mp4")
    for mp4_filename in mp4_filenames:
        basename = mp4_filename.split(".")[0]
        processPerVideo(basename)

if __name__ == "__main__":
    main()


