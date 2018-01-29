import cv2
import dlib
import imutils
from imutils import face_utils
from MouthInfo import MouthInfo

"""
测试开关，打开的时候，在保存n种嘴型信息的时候，会把嘴打印出来
"""
TEST_TYPE_LIPS_MOUTH = True

"""
测试开关，打开的时候，在区分数据是何种嘴型时候，会打印一些信息
"""
TEST_COMPARE_TO_TRAIN_DATA = False

"""
n种嘴型信息对应的人脸图片
"""
LIP_TYPES = [
            "1.png",
            "2.png",
            "3.png",
            "4.png",
            "5.png",
            "6.png",
            "7.png",
            "8.png",
            "9.png",
            "10.png",
            "11.png",
            "12.png",
            "13.png",
            "14.png",
            "15.png",
            "16.png",
            "17.png",
             ]

"""
n种嘴型信息，保存在这个数组里面
"""
MOUTH_TYPE_INFO_ARRAY = []

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def getFacePoint(image):
    #先框出所有的人脸
    rects = detector(image,1)

    #只取第一个脸,因为保证只有一个脸
    if len(rects) == 0:
        return None

    else:
        rect = rects[0]

        #根据框出的人脸，找到人脸所有点（68个点）的坐标
        point = predictor(image,rect)
        point = face_utils.shape_to_np(point)
        return point

def generate112image():
    for index in range(len(LIP_TYPES)):
        image = cv2.imread(LIP_TYPES[index])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = getFacePoint(gray)


        (leftX, leftY) = shape[0]
        (rightX, rightY) = shape[16]
        (topX, topY) = shape[27]
        (bottomX, bottomY) = shape[57]
        finalWidth = (rightX - leftX)
        finalHeight = finalWidth
        centerFaceX_start = leftX
        centerFaceY_start = ((int)((topY + bottomY) / 2 - finalHeight / 2))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = (255,255,255)

        for i in range(len(shape)):
            if(i >= 48 and i <= 67):
                (x,y) = shape[i]
                cv2.circle(image,(x,y), 1, (0,0,0),-1)

        image = image[centerFaceY_start:centerFaceY_start + finalHeight, centerFaceX_start:centerFaceX_start + finalWidth]
        try:
            if image.shape[0] == 0 or image.shape[1] == 0:
                image = None
            else:
                image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
        except ZeroDivisionError:
            image = None
        if image is None:
            pass
        else:
            outputFileName = "{:04d}.jpg".format(index)
            cv2.imshow("generate",image)
            cv2.imwrite(outputFileName, image)


def initTypeMouthInfo(index):
    image = cv2.imread(LIP_TYPES[index])
    # image = imutils.resize(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    shape = getFacePoint(gray)

    #将嘴部信息保存起来
    temp = MouthInfo(shape)
    MOUTH_TYPE_INFO_ARRAY.append(temp)

    #可视化测试代码，批量处理时要关掉
    if TEST_TYPE_LIPS_MOUTH:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i,j] = (255,255,255)

        for i in range(len(shape)):
            if(i >= 48 and i <= 67):
                (x,y) = shape[i]
                cv2.circle(image,(x,y), 3, (0,0,0),-1)
        # image = image[temp.getOffsetY():(temp.getOffsetY()+temp.getHeight()), temp.getOffsetX():(temp.getOffsetX() +temp.getWidth())]
        image = cutImageByMouthRegion(image, temp)
        cv2.imshow("Output",image)
        print("point %d" % index)
        temp.toString()
        cv2.waitKey(0)


def cutImageByMouthRegion(image, mouthInfo):
    image = image[
                    mouthInfo.getOffsetY():(mouthInfo.getOffsetY()+ mouthInfo.getHeight()),
                    mouthInfo.getOffsetX():(mouthInfo.getOffsetX() +mouthInfo.getWidth())
                 ]
    return image


"""
初始化n种嘴型的信息。后面归类嘴型，要与这里的信息比对
"""
def initAllTypeMouth():
    for i in range(len(LIP_TYPES)):
        initTypeMouthInfo(i)

def clasifyMouthType(imgName):
    image = cv2.imread(imgName)
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    minIndex = -1;
    minDistance = 100000;
    errorHappened = False

    for i in range(len(MOUTH_TYPE_INFO_ARRAY)):
        mouthInfo = MOUTH_TYPE_INFO_ARRAY[i]
        comparedPoint = getFacePoint(gray)
        # 没有识别出人脸不往下执行
        if comparedPoint is None:
            errorHappened = True
            break

        imageMouthWidth = MouthInfo.getMouthWidth(comparedPoint)
        imageWidth = image.shape[1]

        #先把嘴的宽度缩放至和要比对的嘴一致,再计算两个嘴之前的点距离
        imageNeedExpand = int(mouthInfo.getWidth() * imageWidth / imageMouthWidth)
        expandImage = imutils.resize(gray,width = imageNeedExpand)
        expandPoint = getFacePoint(expandImage)
        if expandPoint is None:
            errorHappened = True
            break

        expandMouthInfo = MouthInfo(expandPoint)

        if(TEST_COMPARE_TO_TRAIN_DATA):
            mouthImage = cutImageByMouthRegion(expandImage,expandMouthInfo)
            cv2.imshow("expandImage",mouthImage)
            cv2.waitKey(0)

        distance = expandMouthInfo.compareDistance(mouthInfo)
        print("[info] compare with %d, distace %d" % (i,distance))

        if(TEST_COMPARE_TO_TRAIN_DATA):
            print("[info]mouth width:(type, compare):(%d,%d)" % (mouthInfo.getWidth(), expandMouthInfo.getWidth()))
            print("type")
            mouthInfo.toString()
            print("compare")
            expandMouthInfo.toString()

        #取出距离最小的
        if distance < minDistance:
            minDistance = distance
            minIndex = i

    if errorHappened:
        return -1
    else:
        return minIndex

if __name__ == "__main__":
    initAllTypeMouth()
