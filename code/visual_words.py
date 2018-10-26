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
import numpy as np

def image_prepo(image):
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

	return image

def get_gaussian_response(image, sigma, order1):
	image = image.copy()
	r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
	nimg = image
	r = scipy.ndimage.gaussian_filter(r, sigma, order = order1, mode='constant')
	g = scipy.ndimage.gaussian_filter(g, sigma, order = order1, mode='constant')
	b = scipy.ndimage.gaussian_filter(b, sigma, order = order1, mode='constant')
	nimg[:, :, 0] = r
	nimg[:, :, 1] = g
	nimg[:, :, 2] = b
	return nimg

def get_gaussian_laplace_response(image, sigma):
	image=image.copy()
	r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
	nimg = image
	r = scipy.ndimage.gaussian_laplace(r, sigma, mode='constant')
	g = scipy.ndimage.gaussian_laplace(g, sigma, mode='constant')
	b = scipy.ndimage.gaussian_laplace(b, sigma, mode='constant')
	nimg[:, :, 0] = r
	nimg[:, :, 1] = g
	nimg[:, :, 2] = b
	return nimg



def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	sigma1 = (2, 4, 8, 8 * np.sqrt(2))
	sigma2 = (1, 2, 4, 8, 8 * np.sqrt(2))

	x = get_gaussian_response(image, 1, order1=[0, 0])
	for i in sigma2:
		if i > 1:
			a = get_gaussian_response(image, i, order1=[0, 0])
			a = np.concatenate((x, a), axis = 2)
		else:
			a = x
		b = get_gaussian_laplace_response(image, i)
		ab = np.concatenate((a, b), axis=2)
		c = get_gaussian_response(image, i, order1=[1, 0])
		abc = np.concatenate((ab, c), axis=2)
		d = get_gaussian_response(image, i, order1=[0, 1])
		x = np.concatenate((abc, d), axis=2)


	return x


def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	
	image = image_prepo(image)
	wordmap = np.ones(image.shape[0]*image.shape[1]).astype(int)
	filter_responses = extract_filter_responses(image)
	filter_responses_flattern = filter_responses.reshape((-1, 60))
	distance = scipy.spatial.distance.cdist(filter_responses_flattern[0,:].reshape((1,60)), dictionary)

	for i in range(filter_responses_flattern.shape[0]):
		distance = scipy.spatial.distance.cdist(filter_responses_flattern[i,:].reshape((1,60)), dictionary)
		wordmap[i] = int(np.where(distance == np.min(distance))[1][0])

	wordmap = wordmap.reshape((image.shape[0],image.shape[1]))
	return wordmap


def compute_dictionary_one_image(i, alpha, image_path):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''
	image = skimage.io.imread(image_path)
	image = image_prepo(image)
	filter_responses = extract_filter_responses(image)
	filter_responses_flattern = filter_responses.reshape((-1, 60))

	b = np.random.randint(0, filter_responses_flattern.shape[0], alpha)
	c = filter_responses_flattern[b]

	np.save("../data/temporary/%s.npy" % (i), c)


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''
	alpha = 50
	K = 100
	processes = []
	train_data = np.load("../data/train_data.npz")
	path = "../data"
	x = 0
	for i, file in enumerate(train_data['image_names']):
		image_path = path + '/' + file[0]
		p = mp.Process(target=compute_dictionary_one_image, args=(i, alpha, image_path,))
		p.start()
		processes.append(p)
		if i % num_workers==0:
			for p in processes:
				p.join()

	print("filtering done")

	train_data = np.load("../data/train_data.npz")


	results = np.zeros((train_data['image_names'].shape[0]*alpha,60))

	path = "../data/temporary"
	files = os.listdir(path)
	bb = 0
	for i, file in enumerate(files):

		if os.path.splitext(file)[1] == ".npy":
			npy_path = path + '/' + file
			tmp = np.load(npy_path)
			results[bb*alpha:(bb+1)*alpha,:] = tmp
			bb = bb+1


	np.save("../data/train1.npy", results)
	print("Finish storing the training dataset")

	results = np.load("../data/train1.npy")
	kmeans = sklearn.cluster.KMeans(n_clusters = K).fit(results)
	dictionary = kmeans.cluster_centers_
	np.save("../data/dictionary.npy", dictionary)
	print(dictionary.shape)

