# -*- coding: utf-8 -*-
# @File       : 1_generate file info.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-24 19:32
# @Description:

import os
import sys
from os.path import dirname, abspath

path = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(path)
print(path)
output_path = "../../pipeline/2_generate file info"

files = os.listdir("../../DeepFaceLab-master/workspace/src_img_face_512")
# files = [path.replace("\\","/") + f'/Real_ESRGAN_master/datasets/face_512/{w}\n' for w in files]
files = [f'E:/WorkSpace/python/data/face_512/{w}\n' for w in files]
with open(output_path + "/meta_info.txt","w") as f:
    f.writelines(files)
    f.close()
