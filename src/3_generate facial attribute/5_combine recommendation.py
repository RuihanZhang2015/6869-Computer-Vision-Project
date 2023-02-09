# -*- coding: utf-8 -*-
# @File       : 5_combine recommendation.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-05 15:50
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

name = "mengke"
gender = "Woman"
race = "asian"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"
src_path = f"{output_path}\\face_lib"

dta_1 = pd.read_csv(os.path.join(output_path, name, "face_feature1.csv"))
dta_1['source'] = "head1"
dta_2 = pd.read_csv(os.path.join(output_path, name, "face_feature2.csv"))
dta_2['source'] = "head2"
dta_3 = pd.read_csv(os.path.join(output_path, name, "face_feature3.csv"))
dta_3['source'] = "head3"

dta = pd.concat([dta_1, dta_2, dta_3])
dta = dta[dta['Gender']==gender]
# dta = dta[dta['Race']==race]
dta = dta.sort_values(by=['Sim'], ascending=True)
dta = dta.reset_index(drop=True)
dta['rank'] = dta.index

if not os.path.exists(os.path.join(output_path, name, "recommended4")):
    os.mkdir(os.path.join(output_path, name, 'recommended4'))

for index, item in dta.head(10).iterrows():
    shutil.copy(os.path.join(src_path, item['source'], item['Name']), f"{output_path}\\{name}\\recommended4\\{str(item['rank'])}.jpg")
