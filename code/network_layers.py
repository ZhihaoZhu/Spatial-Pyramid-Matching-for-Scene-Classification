import numpy as np
import scipy.ndimage
import os,time
from scipy import ndimage
import skimage.transform

def extract_deep_feature(x,vgg16_weights):
    '''
    	Extracts deep features from the given VGG-16 weights.

    	[input]
    	* x: numpy.ndarray of shape (H,W,3)
    	* vgg16_weights: numpy.ndarray of shape (L,3)

    	[output]
    	* feat: numpy.ndarray of shape (K)
    	'''
    x = x.astype('float') / 255
    x = skimage.transform.resize(x, (224, 224))
    mean = [0.485, 0.456, 0.406]
    mean = np.array(mean).reshape((1,1,3))
    std = [0.229, 0.224, 0.225]
    std = np.array(mean).reshape((1,1,3))
    x = (x-mean)/std
    linear_index = 0
    for i in range(len(vgg16_weights)):
        if vgg16_weights[i][0] == 'conv2d':
            weight = vgg16_weights[i][1]
            bias = vgg16_weights[i][2]
            x = multichannel_conv2d(x,weight,bias)
            # print("conv")
            # print(x.shape)
        elif vgg16_weights[i][0] == 'relu':
            x = relu(x)
            # print("relu")
            # print(x.shape)
        elif vgg16_weights[i][0] == 'maxpool2d':
            size = vgg16_weights[i][1]
            x = max_pool2d(x,size)
            # print("maxpool")
            # print(x.shape)
        else:
            linear_index = linear_index+1
            if linear_index == 3:
                break
            else:
                weight = vgg16_weights[i][1]
                bias = vgg16_weights[i][2]
                x = linear(x, weight, bias)
                # print("linear")
                # print(x.shape)
    return x


def multichannel_conv2d(x,weight,bias):
    '''
    	Performs multi-channel 2D convolution.

    	[input]
    	* x: numpy.ndarray of shape (H,W,input_dim)
    	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
    	* bias: numpy.ndarray of shape (output_dim)

    	[output]
    	* feat: numpy.ndarray of shape (H,W,output_dim)
    '''

    K = weight.shape[1]
    J = weight.shape[0]
    H, W = x.shape[0], x.shape[1]
    weight = np.flip(weight, (2,3))
    print(K,J,H,W)
    y = np.zeros((H, W, J))
    for j in range(J):
        q = np.zeros((H, W))
        for k in range(K):
            p = ndimage.convolve(x[:, :, k], weight[j, k, :, :], mode='constant', cval=0.0)
            q = q + p
        y[:, :, j] = q + bias[j]

    return y

def relu(x):
    '''
    	Rectified linear unit.

    	[input]
    	* x: numpy.ndarray

    	[output]
    	* y: numpy.ndarray
    '''

    shape = x.shape

    x = x.reshape(-1)
    x = np.array([*map(relu_add, x)]).reshape(shape)        #relu_add function is defined just below

    return x

def relu_add(x):
    if x>=0:
        return x
    else:
        return 0

def max_pool2d(x,size):
    '''
    	2D max pooling operation.

    	[input]
    	* x: numpy.ndarray of shape (H,W,input_dim)
    	* size: pooling receptive field

    	[output]
    	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
    	'''
    J = x.shape[2]
    H = int(x.shape[0]/size)
    W = int(x.shape[1]/size)
    y = np.zeros((H,W,J))
    for p in range(H):
        for q in range(W):
            matrix = x[p:(p+1)*size, q:(q+1)*size, :].reshape((-1,J))
            matrix = np.transpose(matrix)
            matrix = np.max(matrix, axis=1)
            y[p,q,:] = matrix
    return y

def linear(x,W,b):
    '''
    	Fully-connected layer.
    	[input]
    	* x: numpy.ndarray of shape (input_dim)
    	* weight: numpy.ndarray of shape (output_dim,input_dim)
    	* bias: numpy.ndarray of shape (output_dim)

    	[output]
    	* y: numpy.ndarray of shape (output_dim)
    	'''
    try:
        x = x.transpose(2, 0, 1)
    except:
        pass
    x = x.reshape((-1))
    y = np.dot(W, x) + b
    return y




