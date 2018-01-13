MOUTH_LEFT_INDEX = 48
MOUTH_RIGHT_INDEX = 54
MOUTH_TOP_INDEX = [50, 51, 52]
MOUTH_BOTTOM_INDEX = 57

MOUTH_START_POINT = 48
MOUTH_END_POINT = 67

"""
图片经人脸识别后，要再保存嘴部的相关坐标，
但这里坐标要变化下，以嘴部左上为(0,0)点。
由于只能先识别人脸，再标出嘴部，so，只能先这样做了。
"""
class MouthInfo(object):
    def __init__(self, faceShape):
        #嘴型的宽,后面比对的时候，对比方要先把缩放至和这个同样大小
        self.width = faceShape[MOUTH_RIGHT_INDEX][0] - faceShape[MOUTH_LEFT_INDEX][0]

        #取左边的偏移量
        offsetLeftX = faceShape[MOUTH_LEFT_INDEX][0]

        #取三个点中最上面的,做为上面的偏移量
        offsetTopY = faceShape[MOUTH_BOTTOM_INDEX][1]
        for topIndex in MOUTH_TOP_INDEX:
            if faceShape[topIndex][1] < offsetTopY:
                offsetTopY = faceShape[topIndex][1]

        self.height = faceShape[MOUTH_BOTTOM_INDEX][1] - offsetTopY

        self.offsetX = offsetLeftX
        self.offsetY = offsetTopY

        self.shapeArray = []
        #保护嘴部的信息，要减去左边和上边的偏移，做为该种嘴型的信息, 比对时进行相应的座标值差
        for i in range(len(faceShape)):
            if(i >= MOUTH_START_POINT and i <= MOUTH_END_POINT):
                (x,y) = faceShape[i]
                x = x - offsetLeftX
                y = y - offsetTopY
                self.shapeArray.append((x,y))

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getXY(self,index):
        return self.shapeArray[index]

    def getOffsetX(self):
        return self.offsetX

    def getOffsetY(self):
        return self.offsetY

    def compareDistance(self, otherMouth):
        sum = 0
        for i in range(len(self.shapeArray)):
            distanceY = self.shapeArray[i][1] - otherMouth.getXY(i)[1]
            distanceX = self.shapeArray[i][0] - otherMouth.getXY(i)[0]
            # (distanceX, distanceY) = self.shapeArray[i] - otherMouth.getXY(i)
            sum = sum + distanceX * distanceX + distanceY * distanceY
        return sum

    @staticmethod
    def getMouthWidth(faceShape):
        return faceShape[MOUTH_RIGHT_INDEX][0] - faceShape[MOUTH_LEFT_INDEX][0]

    def toString(self):
        print("point size:"+str(len(self.shapeArray)))
        for i in range(len(self.shapeArray)):
            print("(%d,%d)" % (self.shapeArray[i][0],self.shapeArray[i][1]), end=" ")
        print("\n")


