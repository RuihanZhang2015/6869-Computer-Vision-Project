# -*- coding: utf-8 -*-
# @File       : 3_evaluate_faceswap_expression.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-08 14:36
# @Description:


import os
from skimage import io
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.metric.GCF import GCF
from src.metric.PSNR import PSNR
from src.metric.EXPRESSION import Expression
from sklearn.metrics import mean_squared_error

index = 1
name = "shangdi"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

result = []

files = os.listdir(f"{output_path}\\{name}\\512_head")
expression = Expression()
for file in tqdm(files):
    raw_img = io.imread(f"{output_path}\\{name}\\512_head\\{file}")
    e1 = expression.get_expression(raw_img)

    swap_img_low = io.imread(f"{output_path}\\{name}\\simswap_{index}_224_head\\{file}")
    e2 = expression.get_expression(swap_img_low)

    swap_img_high = io.imread(f"{output_path}\\{name}\\simswap_{index}_512_head\\{file}")
    e3 = expression.get_expression(swap_img_high)

    temp = {
        "Name": file,
        "Low_Expression": mean_squared_error(e1, e2),
        "High_Expression": mean_squared_error(e1, e3),
    }
    result.append(temp)

result = pd.DataFrame(result)

evaluation = pd.read_csv(os.path.join(output_path,name,"simswap_1_quality_evaluation.csv"))
evaluation = pd.merge(evaluation, result, how="left")
evaluation.to_csv(os.path.join(output_path,name,"simswap_1_quality_evaluation.csv"), index=False)
