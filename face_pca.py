__author__ = 'annapurnaannadatha'
# facial recognition using PCA with Yale face database

from sklearn.decomposition import RandomizedPCA
import numpy
import glob
import cv2
import math
import os
import string
from scipy.misc import *
from scipy import linalg
from skimage.color import rgb2gray
from PIL import Image


img_dims =(200,180)

#image folder path ( Image have been taken from Yale facial database)
path_dim = "img_new/"
path_out = "img_out/"

#function to convert image to right format of grayscale and flatten
def prepare_image(filename):
    #print(filename)
    img_color = cv2.imread(filename)
    img_gray = rgb2gray(img_color)
    return img_gray.flatten()

#PCA using svd
def PCA(x_array):
    mu = numpy.mean(x_array, 0)
    # save mean photo
    imsave(path_out + "mean.jpg",mu.reshape(img_dims))
    # mean adjust the data
    ma_data = x_array - mu
    #ma_data = x_array
    # run SVD
    # eigenvectors , eigenvalues , variance = np. linalg . svd (X.T, full_matrices = False )
    e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)
    # save eigenfaces
    print(e_faces.shape[1])
    for i in range(e_faces.shape[1]):
        data = e_faces[:,i]
        imsave(path_out + str(i) + ".jpg",data.reshape(img_dims))
        #pdb.set_trace()
  	# compute weights for each image
    weights = numpy.dot(ma_data, e_faces)
    return e_faces, weights, mu

# reconstruct an image using the given number of principal components.
def reconstruct(img_idx, e_faces, weights, mu, npcs):
    # reconstruct by dotting weights with the eigenfaces and adding to mean
    #img = mu + numpy.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
    img = mu + numpy.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)


    img_id = str(math.floor(npcs/ 10))+"_" + str(npcs % 10)
    save_image("reconstructed/"+str(img_idx), img_id ,img)

    # dot weights with the eigenfaces and add to mean
    #recon = mu + numpy.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
    return img

#image save
def save_image(out_dir,img_id,data):
	if not os.path.exists(out_dir): os.makedirs(out_dir)
	imsave(out_dir + "/image_" + str(img_id) + ".jpg", data.reshape(img_dims))


# list to store file names of the folder
#faces
listing = os.listdir(path_dim)
'''
# reshape images
for file in listing:
    image = Image.open(path_in + file)
    image = image.resize((img_dims),Image.ANTIALIAS)
    image.save(path_dim + file,optimize=True,quality=95)
'''
#populate into each image into array
#X = numpy.array([prepare_image(path_dim + file) for file in listing])
#for faces
X = numpy.array([imread(path_dim + file,True).flatten() for file in listing])
y = []
for file in listing:
    y.append(file)


# PCA gives eigen faces, weight vector, mu
e_faces, weights, mu = PCA(X)

# reconstruct each face image using an increasing number of principal components
#reconstructed = []
for p in range(X.shape[0]):
    for i in range(X.shape[0]):
        img = reconstruct(p, e_faces, weights, mu, i)
        #reconstructed.append(reconstruct(p, e_faces, weights, mu, i))

print(weights.shape)
print(mu.shape)
print(e_faces.shape)
# perform principal component analysis on the images
pca = RandomizedPCA(n_components =1, whiten=True).fit(X)
X_pca = pca.transform(X)

# scoring phase
test_path = "test/"
test_listing = os.listdir(test_path)
#test files
for file in test_listing:
    Y = numpy.array([prepare_image(test_path + file) ])
    # run through test images
    print(file)
    for j, ref_pca in enumerate(pca.transform(Y)):
        i=0
        distances = []
        # Calculate euclidian distance from test image to each of the known images and save distances
        for i, test_pca in enumerate(X_pca):
            dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
            distances.append((dist,y[i]))
        found_ID = min(distances)[1]
        print("Identified (result: "+ str(found_ID) +" - dist - " + str(min(distances)[0])  + ")")
