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

directory = 'oxbuild_images/'
siftFeatures = 'sift/'

def main():
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
	# Iterate through images.
	for filename in os.listdir(directory):
		# print(filename)
		img = cv2.imread(directory + filename)
		# imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kpl, desc = sift.detectAndCompute(img, None)
		# siftOut = open(siftFeatures + filename[:-4], "w+")
		np.save(siftFeatures + filename[:-4], desc)
		# imgShow = cv2.drawKeypoints(imgGray, kpl, img,
			# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		# plt.imshow(imgShow)
		# plt.show()

		# pdb.set_trace()
		# break;

if __name__ == "__main__":
    main()