# -*- coding: utf-8 -*-
# @File       : 2_match_face_angle.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-24 17:48
# @Description:

import sys
from os.path import dirname, abspath
from sklearn.metrics.pairwise import cosine_similarity

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
sys.path.append(path+"\\metric")
print(path)

import os
import cv2
import pandas
import pandas as pd
from tqdm import tqdm

# from src.metric import POSE_ACC
#
# model = POSE_ACC.Detect_Euler()

# target_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\DeepFaceLab-master\\workspace\\dst_img_face_512\\00001_0.jpg"
target_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\DeepFaceLab-master\\workspace\\src_img_face_512\\00059_0.jpg"
output_path = "../../pipeline/1_detect face angle"

# pic = cv2.imread(f"{target_path}")
# angle = model.detect_euler(pic)
# result = {"Name": "Target",
#             "Type": "dst",
#             "Angle_1": angle[0],
#             "Angle_2": angle[1],
#             "Angle_3": angle[2]}

# Xirui
# result = {
#     "Name": "Target",
#     "Type": "dst",
#     "Angle_1": -1.52855,
#     "Angle_2": -3.340816,
#     "Angle_3": -0.147960
# }

# 第二个女生
result = {
    "Name": "Target",
    "Type": "dst",
    "Angle_1": 0.643866,
    "Angle_2": -2.4643254,
    "Angle_3": 0.56403244
}



dta_face = pd.read_csv(os.path.join(output_path, "face_angle.csv"))
ret = []
for index, item in dta_face.iterrows():
    sim = cosine_similarity([[result['Angle_1'], result['Angle_2'], result['Angle_3']]],
                            [[item['Angle_1'], item['Angle_2'], item['Angle_3']]])
    ret.append(sim[0][0])
dta_face['cos_sim'] = ret
# dta_face = dta_face.sort_values(by=['cos_sim'], ascending=False)
dta_face.to_csv(os.path.join(output_path, "face_angle_2.csv"), index=False)


