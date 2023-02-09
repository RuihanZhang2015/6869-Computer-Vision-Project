# -*- coding: utf-8 -*-
# @File       : 1_get facial attributes.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-03 10:47
# @Description:

import sys
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
sys.path.append(path+"\\metric")
print(path)

import os
import pandas as pd
from tqdm import tqdm
from deepface import DeepFace
backends = ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']

# src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
# dst_path = "../../DeepFaceLab-master/workspace/dst_img_face_128"
src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib\\head3"

# output_path = "../../pipeline/3_generate facial attribute"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib"

src_files = os.listdir(src_path)
# dst_files = os.listdir(dst_path)

ret = []
for file in tqdm(src_files):
    obj = DeepFace.analyze(img_path=f"{src_path}\\{file}",
                       actions = ['gender', 'race'],
                       detector_backend=backends[4],
                       prog_bar=False)
    ret.append({
        "Name": file,
        "Gender": obj['gender'],
        "Race": obj['dominant_race']
    })
ret = pd.DataFrame(ret)
ret.to_csv(os.path.join(output_path, "face_attribute3.csv"), index=False)
