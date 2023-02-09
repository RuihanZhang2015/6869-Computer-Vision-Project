# -*- coding: utf-8 -*-
# @File       : SSIM.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-10 13:53
# @Description:

import numpy as np
from PIL import Image

from skimage.metrics import structural_similarity as ssim

if __name__ == "__main__":
    origin_img = Image.open("img/origin.jpg").convert("L")
    origin_img = np.asarray(origin_img)
    for image in ["blur_face_1.jpg", "blur_face_2.jpg", "blur_face_4.jpg", "blur_face_6.jpg",
                  "blur_all_1.jpg", "blur_all_2.jpg", "blur_all_4.jpg", "blur_all_6.jpg", ]:
        pic = Image.open(image).convert("L")
        pic = np.asarray(pic)
        result = ssim(origin_img, pic,
                      data_range=pic.max() - pic.min())
        print(f"{image}: {result}")
