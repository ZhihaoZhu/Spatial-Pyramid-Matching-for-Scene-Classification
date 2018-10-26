import numpy as np
import multiprocessing as mp
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random
import visual_words

import numpy as np

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''


    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("../data/dictionary.npy")

    layer_num = 3
    dict_size = 100
    features = np.zeros((1440, 2100))
    labels = np.zeros(1440)
    label_dictionary = {'auditorium':0, 'baseball_field':1, 'desert':2, 'highway':3,'kitchen':4, 'laundromat':5,
                        'waterfall':6, 'windmill':7}
    for i, file in enumerate(train_data['image_names']):
        if i % 10 == 0:
            print(i)
        image_path = "../data/"+file[0]
        x = file[0].split('/')[0]
        labels[i] = label_dictionary[x]
        feature = get_image_feature(image_path, dictionary, layer_num, dict_size)
        features[i,:] = feature

    np.savez('../data/trained_system.npz',dictionary = dictionary,features = features,labels = labels,
        SPM_layer_num = layer_num)



def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    label_dictionary = {'auditorium': 0, 'baseball_field': 1, 'desert': 2, 'highway': 3, 'kitchen': 4, 'laundromat': 5,
                        'waterfall': 6, 'windmill': 7}
    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("../data/trained_system.npz")
    conf = np.zeros((8,8))
    features = trained_system['features']
    features_labels = trained_system['labels']
    dictionary = trained_system['dictionary']
    layer_num = trained_system['SPM_layer_num']
    dict_size = 100
    print(features.shape)


    for i, file in enumerate(test_data['image_names']):

        label_name = file[0].split('/')[0]
        true_label = label_dictionary[label_name]
        image_path = "../data/" + file[0]
        feature = get_image_feature(image_path, dictionary, layer_num, dict_size)
        sim = distance_to_set(feature, features)
        predicted_label = int(features_labels[np.where(sim == np.max(sim))[0][0]])
        print(i)
        conf[true_label, predicted_label] = conf[true_label, predicted_label] + 1

    num_accurate = 0
    for i in range(8):
        num_accurate = conf[i,i]+num_accurate

    accuracy = num_accurate/test_data['image_names'].size
    return conf,accuracy


def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K)
    '''
    image = skimage.io.imread(file_path)
    wordmap = visual_words.get_visual_words(image, dictionary)
    feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)


    return feature



def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    dist = np.sum(np.minimum(histograms,word_hist), axis=1)
    return dist




def get_feature_from_wordmap(wordmap,dict_size):

    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    x = wordmap.reshape(-1)
    hist = np.zeros(dict_size)
    for i in range(len(x)):
        hist[x[i]] = hist[x[i]] + 1
    return hist


def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
    '''

    sub_map_length = wordmap.shape[0]//pow(2,layer_num-1)
    sub_map_width = wordmap.shape[1]//pow(2,layer_num-1)
    hist_all = []
    num_pixel = sub_map_length*pow(4,layer_num-1)*sub_map_width
    for i in range(pow(2,layer_num-1)):
        for j in range(pow(2,layer_num-1)):
            wordmap_tbp = wordmap[i*sub_map_length:(i+1)*sub_map_length, j*sub_map_width:(j+1)*sub_map_width]
            hist_all = np.concatenate((hist_all, get_feature_from_wordmap(wordmap_tbp,dict_size)/2))
    p1 = [0, 1, 4, 5]
    p2 = [2, 3, 6, 7]
    p3 = [8, 9, 12, 13]
    p4 = [10, 11, 14, 15]
    P = (p1,p2,p3,p4)
    y = np.zeros(dict_size)

    for p in P:
        x = np.zeros(dict_size)
        for i in p:
            x = hist_all[i * dict_size:(i + 1) * dict_size]*2+x
        y = y+x
        hist_all = np.concatenate((hist_all, x/4))
    hist_all = np.concatenate((hist_all, y/4))

    return hist_all/num_pixel















