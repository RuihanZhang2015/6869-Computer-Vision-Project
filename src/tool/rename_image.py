# -*- coding: utf-8 -*-
# @File       : rename_image.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-06 8:35
# @Description:

import os
src = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data\\shangdi\\2048_head_realesrgan"

files = os.listdir(src)

for file in files:
    if "out" in file:
        new_name = file.replace("_out","")
        os.rename(f"{src}\\{file}", f"{src}\\{new_name}")
