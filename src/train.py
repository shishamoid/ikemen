import dlib
import cv2
import numpy
import torch.nn as nn
import torch
import glob
import os,sys
import re
from torchvision import models, transforms
# -*- coding: utf-8 -*-

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else cpu)
    print("使用デバイス: ",device)
    net.to(device)
    #torch.backends.cundnn.deterministic = True
    torch.backends.cudnn.benchmark = True
