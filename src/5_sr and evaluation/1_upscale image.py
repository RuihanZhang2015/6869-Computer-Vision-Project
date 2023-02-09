# -*- coding: utf-8 -*-
# @File       : 1_upscale image.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-05 19:29
# @Description:

import os
import cv2
from tqdm import tqdm

index = 1
name = "shangdi"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

result = []

# files = os.listdir(f"{output_path}\\{name}\\simswap_{index}_512_head")
files = os.listdir(f"{output_path}\\{name}\\512_head")
out_dim = 2000

# if not os.path.exists(f"{output_path}\\{name}\\simswap_{index}_2048_head_cv2"):
#     os.mkdir(f"{output_path}\\{name}\\simswap_{index}_2048_head_cv2")
#
# for file in tqdm(files[:50]):
#     img = cv2.imread(f"{output_path}\\{name}\\simswap_{index}_512_head\\{file}")
#     img_resize = cv2.resize(img, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
#     cv2.imwrite(f"{output_path}\\{name}\\simswap_{index}_2048_head_cv2\\{file}", img_resize)

if not os.path.exists(f"{output_path}\\{name}\\2048_head_cv2"):
    os.mkdir(f"{output_path}\\{name}\\2048_head_cv2")

for file in tqdm(files[:50]):
    img = cv2.imread(f"{output_path}\\{name}\\512_head\\{file}")
    img_resize = cv2.resize(img, (out_dim, out_dim), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{output_path}\\{name}\\2048_head_cv2\\{file}", img_resize)

