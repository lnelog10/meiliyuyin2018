from imutils import face_utils
import numpy as np
import argparse
import dlib
import cv2
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
# ap.add_argument("-o","--output", required=True, help="path to output video")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["input"] + ".mp4")

print("frame count:", cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("frame rate:", cap.get(cv2.CAP_PROP_FPS))

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

fourcc = cv2.VideoWriter_fourcc(*'XVID')

(x, y, w, h) = (None, None, None, None)

time_count = -1
SLOT_TIME = 250
time_stamp = 0

while (cap.isOpened()):
    time_count = time_count + 1
    #250ms中取中间那一帧
    time_stamp = time_count * SLOT_TIME + 125
    cap.set(cv2.CAP_PROP_POS_MSEC, time_stamp)
    ret, image = cap.read()
    if ret == True:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        print("====================================================>")
        print("big image type:", image.dtype)
        print("gray image type:", gray.dtype)

        #取第一张人脸
        if len(rects) > 0:
            rect = rects[0]
        else:
            continue
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        (x1, y1, w1, h1) = (x, y, w, h)

        (desireHeight, desireWidth) = image.shape[:2]
        desireHeight = desireWidth

        (leftEyeOutX, leftEyeOutY) = shape[36]
        (rightEyeOutX, rightEyeOutY) = shape[45]
        (leftEyeInnerX, leftEyeInnerY) = shape[39]
        (rightEyeInnerX, rightEyeInnerY) = shape[42]

        leftEyeY = (leftEyeOutY + leftEyeInnerY) / 2
        leftEyeX = (leftEyeOutX + leftEyeInnerX) / 2
        rightEyeY = (rightEyeOutY + rightEyeInnerY) / 2
        rightEyeX = (rightEyeOutX + rightEyeInnerX) / 2

        eyesCenterY = ((int)((leftEyeY + rightEyeY) / 2))
        eyesCenterX = ((int)((leftEyeX + rightEyeX) / 2))

        dy = (rightEyeY - leftEyeY)
        dx = (rightEyeX - leftEyeX)
        length = cv2.sqrt(dx * dx + dy * dy)
        angle = math.atan2(dy, dx) * 180.0 / np.pi
        scale = 1

        rows, cols = ((y1 + h1), (x1 + w1))
        M = cv2.getRotationMatrix2D((eyesCenterX, eyesCenterY), angle, 1)
        image = cv2.warpAffine(image, M, (desireWidth, desireHeight))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)

        #只取第一张
        if len(rects) > 0:
            rect = rects[0]
        else:
            continue

        # shape = predictor(gray, rect)
        # shape = face_utils.shape_to_np(shape)
        # (x, y, w, h) = face_utils.rect_to_bb(rect)

        # (leftX, leftY) = shape[0]
        # (rightX, rightY) = shape[16]
        # (topX, topY) = shape[27]
        # (bottomX, bottomY) = shape[57]
        # finalWidth = (rightX - leftX)
        # finalHeight = finalWidth

        # centerFaceX_start = leftX
        # centerFaceY_start = ((int)((topY + bottomY) / 2 - finalHeight / 2))

        # image = image[centerFaceY_start:centerFaceY_start + finalHeight,
        #         centerFaceX_start:centerFaceX_start + finalWidth]

        # try:
        if image.shape[0] == 0 or image.shape[1] == 0:
            image = None
            # else:
            #     image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
        # except ZeroDivisionError:
        #     image = None

        if image is None:
            pass
        else:
            print("time count:", time_count)
            print("final image shape :", image.shape)

            outputFileName = args["input"] + "{:04d}.jpg".format(time_count + 1)
            cv2.imwrite(outputFileName, image)
            # if time_count == 982:
            #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break;

cap.release()
cv2.destroyAllWindows()
