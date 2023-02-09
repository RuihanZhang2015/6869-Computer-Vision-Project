# -*- coding: utf-8 -*-
# @File       : 1_evaluate faceswap.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-05 17:44
# @Description:

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.metric.GCF import GCF
from src.metric.PSNR import PSNR

index = 1
name = "mengke"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

result = []

files = os.listdir(f"{output_path}\\{name}\\frame")
for file in tqdm(files[:50]):
    pic_ori = Image.open(f"{output_path}\\{name}\\frame\\{file}").convert("L")
    pic_ori = np.asarray(pic_ori)

    pic_target_low = Image.open(f"{output_path}\\{name}\\simswap_{index}_224\\{file}").convert("L")
    pic_target_low = np.asarray(pic_target_low)

    pic_target_high = Image.open(f"{output_path}\\{name}\\simswap_{index}_512\\{file}").convert("L")
    pic_target_high = np.asarray(pic_target_high)

    temp = {
        "Name": file,
        "Ori_GCF": GCF(pic_ori),
        "Low_GCF": GCF(pic_target_low),
        "High_GCF": GCF(pic_target_high),
        "Low_PSNR": PSNR(pic_ori, pic_target_low),
        "High_PSNR": PSNR(pic_ori, pic_target_high)
    }
    result.append(temp)

result = pd.DataFrame(result)
result.to_csv(f"{output_path}\\{name}\\simswap_{index}_quality_evaluation.csv", index=False)
print(f"Mean ori GCF: {np.mean(result['Ori_GCF'])}")
print(f"Mean low GCF: {np.mean(result['Low_GCF'])}")
print(f"Mean high GCF: {np.mean(result['High_GCF'])}")
print(f"Mean low PSNR: {np.mean(result['Low_PSNR'])}")
print(f"Mean high PSNR: {np.mean(result['High_PSNR'])}")
