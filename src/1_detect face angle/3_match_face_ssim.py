# -*- coding: utf-8 -*-
# @File       : 3_match_face_ssim.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-01 13:54
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
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize

# target_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\DeepFaceLab-master\\workspace\\dst_img_face_512\\00001_0.jpg"
target_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\DeepFaceLab-master\\workspace\\src_img_face_512\\00059_0.jpg"
src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
output_path = "../../pipeline/1_detect face angle"

dta_face = pd.read_csv(os.path.join(output_path, "face_angle_2.csv"))

ret = []
origin_img = Image.open(target_path).convert("L")
origin_img = np.asarray(origin_img)
origin_img = resize(origin_img, (1024, 1024))

for index, item in tqdm(dta_face.iterrows()):
    pic = Image.open(os.path.join(src_path, item['Name'])).convert("L")
    pic = np.asarray(pic)
    result = ssim(origin_img, pic,
                  data_range=pic.max() - pic.min())
    ret.append(result)
dta_face['ssim'] = ret
dta_face.to_csv(os.path.join(output_path, "face_angle_2.csv"), index=False)
