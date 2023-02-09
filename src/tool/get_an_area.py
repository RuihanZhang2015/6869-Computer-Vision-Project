# -*- coding: utf-8 -*-
# @File       : 2_get an area.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-05 19:10
# @Description:


import cv2

input_file = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\SimSwap\\output\\xirui_out_hd_3.mp4"
output_file = "E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\SimSwap\\output\\xirui_out_hd_3_head.mp4"


cap = cv2.VideoCapture(input_file)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
size = (200, 200)
out = cv2.VideoWriter(output_file, fourcc, fps, size)

# cv2.namedWindow("Input")


while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[250:450,900:1100,:]
    out.write(frame)
    # cv2.imshow("Input", frame)
    # cv2.waitKey(0)


out.release()
cap.release()


