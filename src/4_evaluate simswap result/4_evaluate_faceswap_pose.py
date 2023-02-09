# -*- coding: utf-8 -*-
# @File       : 4_evaluate_faceswap_pose.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-08 14:58
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

result = []

files = os.listdir(f"{output_path}\\{name}\\512_head")
mModel = Detect_Euler()
for file in tqdm(files):
    origin_img = cv2.imread(f"{output_path}\\{name}\\512_head\\{file}")
    origin_angle = mModel.detect_euler(origin_img)

    simswap_img_low = cv2.imread(f"{output_path}\\{name}\\simswap_{index}_224_head\\{file}")
    simswap_angle_low = mModel.detect_euler(simswap_img_low)

    simswap_img_high = cv2.imread(f"{output_path}\\{name}\\simswap_{index}_512_head\\{file}")
    simswap_angle_high = mModel.detect_euler(simswap_img_high)

    sim_low = cosine_similarity([[origin_angle[0], origin_angle[1], origin_angle[2]]],
                            [[simswap_angle_low[0], simswap_angle_low[1], simswap_angle_low[2]]])[0][0]

    sim_high = cosine_similarity([[origin_angle[0], origin_angle[1], origin_angle[2]]],
                                [[simswap_angle_high[0], simswap_angle_high[1], simswap_angle_high[2]]])[0][0]


    temp = {
        "Name": file,
        "Low_Pose": sim_low,
        "High_Pose": sim_high
    }
    result.append(temp)

result = pd.DataFrame(result)

evaluation = pd.read_csv(os.path.join(output_path,name,"simswap_1_quality_evaluation.csv"))
evaluation = pd.merge(evaluation, result, how="left")
evaluation.to_csv(os.path.join(output_path,name,"simswap_1_quality_evaluation.csv"), index=False)
