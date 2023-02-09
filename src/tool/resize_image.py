# -*- coding: utf-8 -*-
# @File       : resize_image.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-04 21:31
# @Description:

import os
import cv2
from tqdm import tqdm

# name = "qing"
# dir_par = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"
# out_dim = 256
#
# if not os.path.exists(os.path.join(dir_par, name, f"head_{out_dim}")):
#     os.mkdir(os.path.join(dir_par, name, f"head_{out_dim}"))
#
# files = os.listdir(os.path.join(dir_par,name,"head_512"))
#
# for file in tqdm(files):
#     img = cv2.imread(os.path.join(dir_par,name,"head_512",file))
#     img_resize = cv2.resize(img, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
#     cv2.imwrite(os.path.join(dir_par, name, f"head_{out_dim}", file), img_resize)

# dir_in = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\stylegan3-main\\out"
dir_in = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib\\head3"
dir_out = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\face_lib\\head4"
out_dim = 512

files = os.listdir(dir_in)

for file in tqdm(files):
    img = cv2.imread(os.path.join(dir_in,file))
    img_resize = cv2.resize(img, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dir_out, file), img_resize)
