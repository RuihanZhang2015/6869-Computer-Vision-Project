# -*- coding: utf-8 -*-
# @File       : get_an_area.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-04 13:01
# @Description:

import cv2

index = 1
name = "shangdi"
output_path = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\data"

input_file = f"{output_path}\\{name}\\simswap_{index}_224.mp4"
output_file = f"{output_path}\\{name}\\simswap_{index}_224_head.mp4"
# input_file = f"{output_path}\\{name}\\simswap_{index}_512.mp4"
# output_file = f"{output_path}\\{name}\\simswap_{index}_512_head.mp4"
# input_file = f"{output_path}\\{name}\\{name}.mp4"
# output_file = f"{output_path}\\{name}\\{name}_512_head.mp4"

location = {
    "qing": [280,1680],
    "shangdi": [430,1730],
    "mengke": [370,1580]
}

cap = cv2.VideoCapture(input_file)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (500, 500)
out = cv2.VideoWriter(output_file, fourcc, fps, size)

# cv2.namedWindow("Input")

index = 0
while index < 50:
    print(index)
    ret, frame = cap.read()
    frame = frame[
            location[name][0]:location[name][0]+500,
            location[name][1]:location[name][1]+500,:]
    out.write(frame)
    index += 1
    # cv2.imshow("Input", frame)
    # cv2.waitKey(0)


out.release()
cap.release()

