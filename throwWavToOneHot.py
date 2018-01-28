import glob
import os
import shutil

import MouthTypeUtil

"""
训练集数据存放目录
"""
TRAIN_DATA = os.path.join("data_set","train")

"""
描扫通过数据处理后的目录
"""
def clasifyLipTypeInDir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        print("[info]transform dir"+dir+"'s data")
        #遍历所有的图片,取出对应的声音数据
        imgFilesPattern = dir + os.path.sep + "*.jpg"
        # imgFilesPath = glob.glob(dir,"*.jpg")
        imgFilesPath = glob.glob(imgFilesPattern)
        for imgFile in imgFilesPath:
            lipType = clasifyWitchLipType(imgFile)
            print("[info]file:"+imgFile+" lipType:"+str(lipType))
            destDir = os.path.join(TRAIN_DATA,str(lipType))

            #新建一个分类的文件夹
            if not os.path.exists(destDir):
                os.makedirs(destDir)

            #将图片移动至对应分类的文件夹下面
            shutil.copy(imgFile,destDir)

            #将与图片同名的文件夹也放到对应分类的文件夹下面
            basename = imgFile.split(".")[0]
            print("voicename:"+basename)
            voiceTxtName = basename + ".txt"
            shutil.copy(voiceTxtName,destDir)
    else:
        print("[error]dir does not exist or not dir")

"""
将声音对应的人脸图片与N种类型的嘴型图片对比，区分出是何种嘴型
"""
def clasifyWitchLipType(imgName):
    # basename = os.path.basename(imgName)
    print("[info]file No."+imgName.split(".")[0])
    type = MouthTypeUtil.clasifyMouthType(imgName)
    return type


def main():
    MouthTypeUtil.initAllTypeMouth()
    dirs = glob.glob("data_pre_process")
    for dir in dirs:
        if os.path.isdir(dir):
            print("[info]:"+dir)
            clasifyLipTypeInDir(dir)

if __name__ == '__main__':
    main()