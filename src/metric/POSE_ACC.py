# -*- coding: utf-8 -*-
# @File       : POSE_ACC.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-10 14:12
# @Description:

import sys
from os.path import dirname, abspath

path = dirname(abspath(__file__))
print(path)
sys.path.append(path)

import cv2
import numpy as np
from keras import backend as K
from lib.FSANET_model import *
from keras.layers import Average

class Detect_Euler:
    def __init__(self):
        K.set_learning_phase(0)  # make sure its testing mode
        self.face_cascade = cv2.CascadeClassifier(path + '/pre-trained/lbpcascade_frontalface_improved.xml')

        # load model and weights
        self.img_size = 64
        stage_num = [3, 3, 3]
        lambda_local = 1
        lambda_d = 1
        img_idx = 0
        detected = ''  # make this not local variable
        time_detection = 0
        time_network = 0
        time_plot = 0
        skip_frame = 5  # every 5 frame do 1 detection and network forward propagation
        self.ad = 0.6

        # Parameters
        num_capsule = 3
        dim_capsule = 16
        routings = 2
        stage_num = [3, 3, 3]
        lambda_d = 1
        num_classes = 3
        image_size = 64
        num_primcaps = 7 * 3
        m_dim = 5
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
        model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

        num_primcaps = 8 * 8 * 3
        S_set = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

        model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

        print('Loading models ...')

        weight_file1 = path + '/pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5'
        model1.load_weights(weight_file1)
        print('Finished loading model 1.')

        weight_file2 = path + '/pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5'
        model2.load_weights(weight_file2)
        print('Finished loading model 2.')

        weight_file3 = path + '/pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5'
        model3.load_weights(weight_file3)
        print('Finished loading model 3.')

        inputs = Input(shape=(self.img_size, self.img_size, 3))
        x1 = model1(inputs)  # 1x1
        x2 = model2(inputs)  # var
        x3 = model3(inputs)  # w/o
        avg_model = Average()([x1, x2, x3])
        self.model = Model(inputs=inputs, outputs=avg_model)

    def get_euler(self, detected, input_img, faces, ad, img_size, img_w, img_h, model):
        result = None
        for i, (x, y, w, h) in enumerate(detected):
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)

            faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))
            faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            face = np.expand_dims(faces[i, :, :, :], axis=0)
            result = model.predict(face)

            result = result.squeeze()
        if result is None:
            result = [np.nan, np.nan, np.nan]
        return result

    def detect_euler(self, p_img):
        img_h, img_w, _ = np.shape(p_img)

        # detect faces using LBP detector
        gray_img = cv2.cvtColor(p_img, cv2.COLOR_BGR2GRAY)
        detected = self.face_cascade.detectMultiScale(gray_img, 1.1)

        faces = np.empty((len(detected), self.img_size, self.img_size, 3))

        euler = self.get_euler(detected, p_img, faces, self.ad, self.img_size, img_w, img_h, self.model)
        return euler


def calculate_distance(p_1, p_2):
    arccos = np.sum(p_1 * p_2) / (np.sqrt(np.sum(p_1**2)) * np.sqrt(np.sum(p_2**2)))
    return np.arccos(arccos) / np.pi * 180


if __name__ == "__main__":
    mModel = Detect_Euler()
    origin_img = cv2.imread("img/origin.jpg")
    origin_angle = mModel.detect_euler(origin_img)
    for image in ["blur_face_1.jpg", "blur_face_2.jpg", "blur_face_4.jpg", "blur_face_6.jpg",
                  "blur_all_1.jpg", "blur_all_2.jpg", "blur_all_4.jpg", "blur_all_6.jpg", ]:
        pic = cv2.imread(path + f"/img/{image}")
        angle = mModel.detect_euler(pic)

        result = calculate_distance(origin_angle, angle)
        print(f"{image}: {result}")
