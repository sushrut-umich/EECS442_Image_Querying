import os
import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

images = 'oxbuild_images/'

def main():
	infile = open('codebook','rb')
	codebook = pickle.load(infile)
	infile.close()
	#codebook = np.load('codebook.npy')
	allDesc = np.load('allDesc.npy')
	# 6. Produce the labels for each of the test samples
	#labels = codebook.predict(X_test)
	#labels2 = codebook.predict(X_train)
	print(allDesc.shape)
	#histogram = np.zeros((5063, 1000))
	labels = codebook.predict(allDesc)
	labels = np.reshape(labels,(5055, 20))

	histogram = np.zeros((5055, 1000))
	for i, label in enumerate(labels):
		for j in label:
			histogram[i][j] += 1
	np.save('histograms', histogram)

if __name__ == "__main__":
	main()