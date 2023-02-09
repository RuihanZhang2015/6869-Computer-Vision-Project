# -*- coding: utf-8 -*-
# @File       : generate meta info.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-04 13:29
# @Description:

import os

input = "E:/WorkSpace/python/data/face_xirui"
output = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\Real_ESRGAN_master\\datasets"

file_name = "meta_info3"

files = os.listdir(input)

with open(f"{output}\\{file_name}.txt", 'w') as f:
    for file in files:
        f.write(f"{input}/{file}\n")
    f.close()
