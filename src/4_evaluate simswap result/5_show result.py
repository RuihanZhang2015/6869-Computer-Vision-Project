# -*- coding: utf-8 -*-
# @File       : 5_show result.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-08 15:05
# @Description:


import os
import cv2
from skimage import io
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.metric.POSE_ACC import Detect_Euler
from sklearn.metrics.pairwise import cosine_similarity

index = 1
name = "shangdi"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

evaluation = pd.read_csv(os.path.join(output_path,name,"simswap_1_quality_evaluation.csv"))
print(f"Mean ori GCF: {np.mean(evaluation['Ori_GCF'])}")
print(f"Mean low GCF: {np.mean(evaluation['Low_GCF'])}")
print(f"Mean high GCF: {np.mean(evaluation['High_GCF'])}")
print(f"Mean low PSNR: {np.mean(evaluation['Low_PSNR'])}")
print(f"Mean high PSNR: {np.mean(evaluation['High_PSNR'])}")
print(f"Mean low Expression: {np.mean(evaluation['Low_Expression'])}")
print(f"Mean high Expression: {np.mean(evaluation['High_Expression'])}")
print(f"Mean low Pose: {np.mean(evaluation['Low_Pose'])}")
print(f"Mean high Pose: {np.mean(evaluation['High_Pose'])}")
