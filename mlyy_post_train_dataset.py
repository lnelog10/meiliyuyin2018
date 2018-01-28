# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Functions for  reading MeiLiYuYin data."""

import os;
import glob as gb
import shutil

Pre_Train_Dir_Path = "./data_set/train"
Post_Train_Dir_Path = "./data_set/train_real"

'读取所有txt文件，并标记它所属的分类_type'
def mark_all_data(train_dir):
    dirs = gb.glob(train_dir + os.sep + '*');
    for p_dir in dirs:
        if os.path.isdir(p_dir):
            post_train_data(p_dir)
        else:
            print("mark_all_data is not dir in root")

def post_train_data(p_dir):
    preTrainPath = p_dir + os.sep + '*.txt'
    voiceFiles = gb.glob(pathname=preTrainPath)
    print(p_dir + " has voices " +str(voiceFiles.__len__()))
    for txt in voiceFiles:
        if os.path.isdir(txt):
            continue
        else:
            p_dir_name = os.path.basename(p_dir)
            voicname = os.path.basename(txt)
            (filename, extension) = os.path.splitext(voicname)
            new_voicname = Post_Train_Dir_Path + os.sep + filename + "_" + p_dir_name +extension;
            shutil.copy(txt, new_voicname)
            print("copy " + txt + " to " + new_voicname)


if __name__ == "__main__":
    mark_all_data(train_dir=Pre_Train_Dir_Path)
