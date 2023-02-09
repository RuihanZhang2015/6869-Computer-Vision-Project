# -*- coding: utf-8 -*-
# @File       : 2_evaluate superresolution.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-06 8:22
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

files = os.listdir(f"{output_path}\\{name}\\2048_head_cv2")
for file in tqdm(files):
    # pic_ori = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_cv2\\{file}").convert("L")
    # pic_ori = np.asarray(pic_ori)
    #
    # pic_mmedit_basicvsrplusplus = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_mmedit_basicvsrplusplus\\{file}").convert("L")
    # pic_mmedit_basicvsrplusplus = np.asarray(pic_mmedit_basicvsrplusplus)
    #
    # pic_mmedit_realbasicvsr = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_mmedit_realbasicvsr\\{file}").convert("L")
    # pic_mmedit_realbasicvsr = np.asarray(pic_mmedit_realbasicvsr)
    #
    # pic_realesrgan = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_realesrgan\\{file}").convert("L")
    # pic_realesrgan = np.asarray(pic_realesrgan)
    #
    # pic_vrt = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_vrt\\{file.replace('jpg','png')}").convert("L")
    # pic_vrt = np.asarray(pic_vrt)

    pic_ori = Image.open(f"{output_path}\\{name}\\2048_head_cv2\\{file}").convert("L")
    pic_ori = np.asarray(pic_ori)

    pic_mmedit_basicvsrplusplus = Image.open(
        f"{output_path}\\{name}\\2048_head_mmedit_basicvsrplusplus\\{file}").convert("L")
    pic_mmedit_basicvsrplusplus = np.asarray(pic_mmedit_basicvsrplusplus)

    pic_mmedit_realbasicvsr = Image.open(
        f"{output_path}\\{name}\\2048_head_mmedit_realbasicvsr\\{file}").convert("L")
    pic_mmedit_realbasicvsr = np.asarray(pic_mmedit_realbasicvsr)

    pic_realesrgan = Image.open(f"{output_path}\\{name}\\2048_head_realesrgan\\{file}").convert("L")
    pic_realesrgan = np.asarray(pic_realesrgan)

    # pic_vrt = Image.open(f"{output_path}\\{name}\\simswap_1_2048_head_vrt\\{file.replace('jpg', 'png')}").convert("L")
    # pic_vrt = np.asarray(pic_vrt)

    temp = {
        "Name": file,
        "Ori_GCF": GCF(pic_ori),
        "mmedit_basicvsrplusplus_GCF": GCF(pic_mmedit_basicvsrplusplus),
        "mmedit_readbasicvsr_GCF": GCF(pic_mmedit_realbasicvsr),
        "realesrgan_GCF": GCF(pic_realesrgan),
        # "vrt_GCF": GCF(pic_vrt)
    }
    result.append(temp)

result = pd.DataFrame(result)
result.to_csv(f"{output_path}\\{name}\\simswap_{index}_sr_evaluation.csv", index=False)
print(f"Mean ori GCF: {np.mean(result['Ori_GCF'])}")
print(f"Mean mmedit basicvsr++ GCF: {np.mean(result['mmedit_basicvsrplusplus_GCF'])}")
print(f"Mean mmedit realbasicvsr GCF: {np.mean(result['mmedit_readbasicvsr_GCF'])}")
print(f"Mean realesrgan GCF: {np.mean(result['realesrgan_GCF'])}")
# print(f"Mean vrt GCF: {np.mean(result['vrt_GCF'])}")
