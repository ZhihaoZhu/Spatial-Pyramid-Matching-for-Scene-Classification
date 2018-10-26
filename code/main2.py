import multiprocessing as mp
import time
import numpy as np
import os
import sklearn.cluster
import scipy
import skimage.io
import skimage.transform
import visual_words
import scipy
import matplotlib.pyplot as plt
import network_layers
import util
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image
import deep_recog
import visual_recog
import network_layer


if __name__ == '__main__':
    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    # conf, accuracy = deep_recog.evaluate_recognition_system(vgg16,2)
    # print(conf)
    # print(accuracy)
    path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    image = skimage.io.imread(path_img)
    vgg16_weights = util.get_VGG16_weights()
    feature_1 = network_layer.extract_deep_feature(image,vgg16_weights)
    print(feature_1)

    feature_2 = deep_recog.get_image_feature(2, path_img, vgg16)
    print(feature_2)




























