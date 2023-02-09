# -*- coding: utf-8 -*-
# @File       : GCF.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-10 10:57
# @Description:

from PIL import Image
import numpy as np
import cv2


def GCF_one_layer(p_image, p_gama=2.2):
    # Step 1: linear luminance
    image_ll = 100 * np.power(p_image / 255, p_gama / 2)

    # Step 2: get local contrast
    temp_contrast = np.zeros((image_ll.shape[0], image_ll.shape[1]), dtype=np.float)
    for shape0 in range(image_ll.shape[0]):
        if shape0 != 0:
            temp_contrast[shape0, :] = temp_contrast[shape0, :] + np.abs(image_ll[shape0, :] - image_ll[shape0 - 1, :])
        if shape0 != image_ll.shape[0] - 1:
            temp_contrast[shape0, :] = temp_contrast[shape0, :] + np.abs(image_ll[shape0, :] - image_ll[shape0 + 1, :])

    for shape1 in range(image_ll.shape[1]):
        if shape1 != 0:
            temp_contrast[:, shape1] = temp_contrast[:, shape1] + np.abs(image_ll[:, shape1] - image_ll[:, shape1 - 1])
        if shape1 != image_ll.shape[1] - 1:
            temp_contrast[:, shape1] = temp_contrast[:, shape1] + np.abs(image_ll[:, shape1] - image_ll[:, shape1 + 1])

    divide = np.ones((image_ll.shape[0], image_ll.shape[1]), dtype=np.float) * 4
    divide[0, :] = 3
    divide[-1, :] = 3
    divide[:, 0] = 3
    divide[:, -1] = 3
    divide[0, 0] = 2
    divide[0, -1] = 2
    divide[-1, 0] = 2
    divide[-1, -1] = 2

    local_contrast = temp_contrast / divide

    # Step 3: Average local contrast
    C = 1 / (image_ll.shape[0] * image_ll.shape[1]) * np.sum(local_contrast)

    return C

def GCF(p_image):
    c = []
    w = []
    for i in range(3):
        w.append((-0.406385*(i+1)/9+0.334573)*(i+1)/9 + 0.0877526)
        c.append(GCF_one_layer(p_image))
        row,col = p_image.shape
        new_row = row // 2
        new_col = col // 2
        p_image = cv2.resize(p_image, (new_row, new_col), interpolation=cv2.INTER_LINEAR)
    ret = 0
    for index in range(len(c)):
        ret += c[index] * w[index]
    return ret


if __name__ == "__main__":
    for image in ["origin.jpg",
                  "blur_face_1.jpg", "blur_face_2.jpg", "blur_face_4.jpg", "blur_face_6.jpg",
                  "blur_all_1.jpg", "blur_all_2.jpg", "blur_all_4.jpg", "blur_all_6.jpg", ]:
        pic = Image.open(f"img//{image}").convert("L")
        pic = np.asarray(pic)
        result = GCF(pic)
        print(f"{image}: {result}")
