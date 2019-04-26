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

directory = 'sift/'

def main():
	firstFile = os.listdir(directory)[0]
	desc = np.load(directory + firstFile)
	size = 200 if desc.shape[0] < 200 else desc.shape[0] - 1
	desc = desc[np.random.randint(desc.shape[0], size=size), :]
	
	allDesc = desc

	for filename in os.listdir(directory)[1:]:
		desc = np.load(directory + filename)
		size = 200 if desc.shape[0] < 200 else desc.shape[0] - 1
		desc = desc[np.random.randint(desc.shape[0], size=size), :]
		# pdb.set_trace()
		allDesc = np.vstack((allDesc, desc))
	pdb.set_trace()
	codebook = KMeans(n_clusters=10000, max_iter=10, verbose=1).fit(allDesc)


if __name__ == "__main__":
    main()