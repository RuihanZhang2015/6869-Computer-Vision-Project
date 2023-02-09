# -*- coding: utf-8 -*-
# @File       : 4_give recommendation.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-03 20:48
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

name = "qing"
gender = "Woman"
race = "asian"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"
src_path = f"{output_path}\\face_lib\\head3"

dta = pd.read_csv(os.path.join(output_path, name, "face_feature3.csv"))
dta = dta[dta['Gender']==gender]
# dta = dta[dta['Race']==race]
dta = dta.sort_values(by=['Sim'], ascending=True)
dta = dta.reset_index(drop=True)
dta['rank'] = dta.index

if not os.path.exists(os.path.join(output_path, name, "recommended3")):
    os.mkdir(os.path.join(output_path, name, 'recommended3'))

for index, item in dta.head(10).iterrows():
    shutil.copy(os.path.join(src_path, item['Name']), f"{output_path}\\{name}\\recommended3\\{str(item['rank'])}.jpg")
