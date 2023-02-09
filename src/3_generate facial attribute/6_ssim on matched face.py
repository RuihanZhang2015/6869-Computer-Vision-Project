# -*- coding: utf-8 -*-
# @File       : 6_ssim on matched face.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-05 17:01
# @Description:

import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

name = "xirui"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

origin_img = Image.open(os.path.join(output_path,name,"head_512\\00001_0.jpg")).convert("L")
origin_img = np.asarray(origin_img)

for image in ["0.jpg", "1.jpg", "2.jpg", "3.jpg","4.jpg"]:
    pic = Image.open(os.path.join(output_path,name,"recommended4", image)).convert("L")
    pic = np.asarray(pic)
    result = ssim(origin_img, pic,
                  data_range=pic.max() - pic.min())
    print(f"{image}: {result}")
