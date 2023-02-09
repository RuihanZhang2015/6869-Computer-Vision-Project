# -*- coding: utf-8 -*-
# @File       : EXPRESSION.py
# @Author     : Yuchen Chai
# @Date       : 2022-05-08 14:01
# @Description:

import sys
from os.path import dirname, abspath

path = dirname(abspath(__file__))
print(path)
sys.path.append(path)
sys.path.append(dirname(path))

import os
import torch
import numpy as np
from PIL import Image
from expression import transforms as transforms
from expression.models import *
from skimage import io
from skimage.transform import resize


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class Expression():
    def __init__(self):
        self.cut_size = 44
        self.transform_test = transforms.Compose([
            transforms.TenCrop(self.cut_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        self.net = VGG('VGG19')
        checkpoint = torch.load('E:\\Dropbox (MIT)\\workspace\\MIT\\Course\\6.869 Computer vision\\project\\src\\metric\\expression\\FER2013_VGG19\\PrivateTest_model.t7')
        self.net.load_state_dict(checkpoint['net'])
        self.net.cuda()
        self.net.eval()

    def get_expression(self, raw_img):
        gray = rgb2gray(raw_img)
        gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
        img = gray[:, :, np.newaxis]

        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        inputs = self.transform_test(img)

        ncrops, c, h, w = np.shape(inputs)

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.cuda()
        inputs = Variable(inputs, volatile=True)
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        score = score.cpu().detach().numpy()
        return score


if __name__ == "__main__":
    raw_img = io.imread('img/origin.jpg')
    expression = Expression()
    expression.get_expression(raw_img)
