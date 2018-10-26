import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import torchvision.transforms as transforms
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
from PIL import Image

def build_recognition_system(vgg16,num_workers=2):
    '''
    	Creates a trained recognition system by generating training features from all training images.

    	[input]
    	* vgg16: prebuilt VGG-16 network.
    	* num_workers: number of workers to process in parallel

    	[saved]
    	* features: numpy.ndarray of shape (N,K)
    	* labels: numpy.ndarray of shape (N)
    	'''

    train_data = np.load("../data/train_data.npz")
    features = np.zeros((1440, 4096))
    labels = np.zeros(1440)
    label_dictionary = {'auditorium': 0, 'baseball_field': 1, 'desert': 2, 'highway': 3, 'kitchen': 4, 'laundromat': 5,
                        'waterfall': 6, 'windmill': 7}
    print(time.asctime(time.localtime(time.time())))
    for i, file in enumerate(train_data['image_names']):
        if i % 10 == 0:
            print(i)
        x = file[0].split('/')[0]
        labels[i] = label_dictionary[x]
        image_path = "../data/"+file[0]
        feature = get_image_feature(i, image_path, vgg16)
        features[i,:] = feature.detach().numpy()[0]

    print(time.asctime(time.localtime(time.time())))
    np.savez('../data/trained_system_deep.npz', features = features, labels = labels)

def evaluate_recognition_system(vgg16,num_workers=2):
    '''
    	Evaluates the recognition system for all test images and returns the confusion matrix.

    	[input]
    	* vgg16: prebuilt VGG-16 network.
    	* num_workers: number of workers to process in parallel

    	[output]
    	* conf: numpy.ndarray of shape (8,8)
    	* accuracy: accuracy of the evaluated system
    	'''
    label_dictionary = {'auditorium': 0, 'baseball_field': 1, 'desert': 2, 'highway': 3, 'kitchen': 4, 'laundromat': 5,
                        'waterfall': 6, 'windmill': 7}
    test_data = np.load("../data/test_data.npz")
    deep_trained_system = np.load("../data/trained_system_deep.npz")
    conf = np.zeros((8, 8))
    features = deep_trained_system['features']
    features_labels = deep_trained_system['labels']

    for i, file in enumerate(test_data['image_names']):

        label_name = file[0].split('/')[0]
        true_label = label_dictionary[label_name]
        image_path = "../data/" + file[0]
        feature = get_image_feature(i, image_path, vgg16).detach().numpy()
        print(type(feature))

        sim = distance_to_set(feature, features)
        predicted_label = int(features_labels[np.where(sim == np.max(sim))[0][0]])
        conf[true_label, predicted_label] = conf[true_label, predicted_label] + 1

    num_accurate = 0
    for i in range(8):
        num_accurate = conf[i, i] + num_accurate

    accuracy = num_accurate / test_data['image_names'].size
    return conf, accuracy

def preprocess_image(image):
    '''
    	Preprocesses the image to load into the prebuilt network.

    	[input]
    	* image: numpy.ndarray of shape (H,W,3)

    	[output]
    	* image_processed: torch.Tensor of shape (3,H,W)
    	'''
    image = image.astype('float') / 255
    if image.ndim == 2:
        nimage = np.zeros((image.shape[0], image.shape[1], 3))
        nimage[:, :, 0] = image
        nimage[:, :, 1] = image
        nimage[:, :, 2] = image
        image = nimage
    elif image.shape[2] == 4:
        nimage = np.zeros((image.shape[0], image.shape[1], 3))
        nimage[:, :, 0] = image[:, :, 0]
        nimage[:, :, 1] = image[:, :, 1]
        nimage[:, :, 2] = image[:, :, 2]
        image = nimage
    x = skimage.transform.resize(image, (224, 224))
    mean = [0.485, 0.456, 0.406]
    mean = np.array(mean).reshape((1, 1, 3))
    std = [0.229, 0.224, 0.225]
    std = np.array(mean).reshape((1, 1, 3))
    x = (x - mean) / std

    image = np.transpose(x, (2,0,1))[np.newaxis,:,:,:]

    tensor = torch.from_numpy(image)

    return tensor


def get_image_feature(i, image_path, vgg16):
    '''
            Extracts deep features from the prebuilt VGG-16 network.
            This is a function run by a subprocess.
             [input]
            * i: index of training image
            * image_path: path of image file
            * vgg16: prebuilt VGG-16 network.
            * time_start: time stamp of start time
             [saved]
            * feat: evaluated deep feature
            '''

    image = skimage.io.imread(image_path)
    tensor = preprocess_image(image)

    classifier = vgg16.classifier
    classifier_modified = torch.nn.Sequential(*list(classifier.children())[0:5])
    vgg16.classifier = classifier_modified
    feature = vgg16(tensor)

    return feature


def distance_to_set(feature,train_features):
    '''
    	Compute distance between a deep feature with all training image deep features.

    	[input]
    	* feature: numpy.ndarray of shape (K)
    	* train_features: numpy.ndarray of shape (N,K)

    	[output]
    	* dist: numpy.ndarray of shape (N)
    	'''
    dist = np.zeros(train_features.shape[0])
    for i in range(train_features.shape[0]):
        dist[i] = -np.linalg.norm(feature - train_features[i])

    dist = np.array(dist)
    return dist

