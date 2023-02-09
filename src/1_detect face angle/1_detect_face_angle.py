# -*- coding: utf-8 -*-
# @File       : 1_detect face angle.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-22 13:10
# @Description:

import sys
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
sys.path.append(path+"\\metric")
print(path)

import os
import cv2
import pandas
import pandas as pd
from tqdm import tqdm

from src.metric import POSE_ACC

model = POSE_ACC.Detect_Euler()

src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
# dst_path = "../../DeepFaceLab-master/workspace/dst_img_face_128"
output_path = "../../pipeline/1_detect face angle"

src_files = os.listdir(src_path)
# dst_files = os.listdir(dst_path)

ret = []
for file in tqdm(src_files):
    pic = cv2.imread(f"{src_path}/{file}")
    angle = model.detect_euler(pic)
    ret.append({"Name": file,
                "Type": "src",
                "Angle_1": angle[0],
                "Angle_2": angle[1],
                "Angle_3": angle[2],})
ret = pd.DataFrame(ret)
ret.to_csv(os.path.join(output_path, "face_angle.csv"), index=False)
