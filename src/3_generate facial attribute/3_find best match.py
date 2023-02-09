# -*- coding: utf-8 -*-
# @File       : 3_find best match.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-03 20:34
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
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
from src.algorithm.LM import Face_Landmark

# src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
# dst_path = "../../DeepFaceLab-master/workspace/dst_img_face_512"

name = "mengke"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"
pickle_path = ""
files = os.listdir(f"{output_path}\\{name}\\head_512")
face_attribute = pd.read_csv(f"{output_path}\\face_lib\\face_attribute1.csv")

face_landmark = Face_Landmark()
# src_files = os.listdir(src_path)

ret = {}
# img = cv2.imread(f"../../DeepFaceLab-master/workspace/src_img_face_512/00100_0.jpg")
img = cv2.imread(f"{output_path}\\{name}\\head_512\\{files[0]}")

marks = face_landmark.get_landmark(img, False)
features = face_landmark.calculate_feature(marks)

with open(os.path.join(output_path, "face_lib\\face_feature1.pickle"),'rb') as f:
    ori_features = pickle.load(f)

def get_similarity(feat_a, feat_b):
    sim = np.sqrt(np.sum(np.power(feat_a - feat_b,2)))
    return sim

simlarity = []
for file in ori_features:
    sim = get_similarity(features, ori_features[file])
    simlarity.append({
        "Name": file,
        "Sim": sim
    })
simlarity = pd.DataFrame(simlarity)
simlarity = pd.merge(simlarity, face_attribute, how="left")
simlarity.to_csv(os.path.join(output_path, name, "face_feature1.csv"), index=False)

