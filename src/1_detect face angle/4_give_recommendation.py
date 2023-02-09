# -*- coding: utf-8 -*-
# @File       : 4_give_recommendation.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-01 14:08
# @Description:

import sys
from os.path import dirname, abspath

path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
sys.path.append(path+"\\metric")
print(path)

import os
import shutil
import pandas as pd
output_path = "../../pipeline/1_detect face angle"
src_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
image_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\DeepFaceLab-master\\workspace\\recommended2\\"

dta = pd.read_csv(os.path.join(output_path, "face_angle_2.csv"))
ratio = max(dta['cos_sim']) / max(dta['ssim'])
dta['final_score'] = 1 * dta['cos_sim'] + ratio * dta['ssim']
dta = dta.sort_values(by=['final_score'], ascending=False)
dta = dta.reset_index(drop=True)
dta['rank'] = dta.index

for index, item in dta.head(10).iterrows():
    shutil.copy(os.path.join(src_path, item['Name']), f"{image_path}\\{str(item['rank'])}.jpg")
