# flake8: noqa
import os
import sys
from os.path import dirname, abspath

path = abspath(__file__)
sys.path.append(path)
sys.path.append(dirname(path))
sys.path.append(dirname(dirname(path)))
sys.path.append(dirname(dirname(dirname(path))))
import os.path as osp
from basicsr.train import train_pipeline

import Real_ESRGAN_master.realesrgan.archs
import Real_ESRGAN_master.realesrgan.data
import Real_ESRGAN_master.realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    print(root_path)
    train_pipeline(root_path)
