# -*- coding: utf-8 -*-
# @File       : test.py
# @Author     : Yuchen Chai
# @Date       : 2022-04-25 12:54
# @Description:

import torch

print(torch.cuda.is_available())
print(torch.version.cuda)

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
