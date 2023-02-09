# -*- coding: utf-8 -*-
# @File       : PSNR.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-10 14:04
# @Description:

from math import log10, sqrt
import numpy as np
from PIL import Image


def PSNR(original, changed):
    mse = np.mean((original - changed) ** 2)
    if mse == 0:
        return 100
    max_pixel = changed.max()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


if __name__ == "__main__":
    origin_img = Image.open("img/origin.jpg").convert("L")
    origin_img = np.asarray(origin_img)
    for image in ["blur_face_1.jpg", "blur_face_2.jpg", "blur_face_4.jpg", "blur_face_6.jpg",
                  "blur_all_1.jpg", "blur_all_2.jpg", "blur_all_4.jpg", "blur_all_6.jpg", ]:
        pic = Image.open(image).convert("L")
        pic = np.asarray(pic)
        result = PSNR(origin_img, pic)
        print(f"{image}: {result}")
