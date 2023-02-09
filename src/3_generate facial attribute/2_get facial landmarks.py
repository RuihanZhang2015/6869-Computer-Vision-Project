# -*- coding: utf-8 -*-
# @File       : 2_get facial landmarks.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-03 20:17
# @Description:


import sys
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
sys.path.append(path+"\\metric")
sys.path.append(path+"\\algorithm")
print(path)

import os
import cv2
import pickle
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
from src.algorithm.LM import Face_Landmark

# src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib\\head3"
# dst_path = "../../DeepFaceLab-master/workspace/dst_img_face_128"
# output_path = "../../pipeline/3_generate facial attribute"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib"

face_landmark = Face_Landmark()
src_files = os.listdir(src_path)

ret = {}
for file in tqdm(src_files):
    img = cv2.imread(f"{src_path}/{file}")
    marks = face_landmark.get_landmark(img, False)
    features = face_landmark.calculate_feature(marks)
    ret[file] = features

with open(os.path.join(output_path, "face_feature3.pickle"),'wb') as f:
    pickle.dump(ret, f)
