import os
import pdb
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

directory = 'oxbuild_images/'
groundtruth = 'gt_files_170407/'

results = 'results/'

def main():
	sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
	# Iterate through images.
	#for filename in os.listdir(directory):
	# print(filename)
	img = cv2.imread(directory + 'all_souls_000013.jpg')
	
	
	img = img[34:956,136:648]
	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#plt.imshow(img)
	#plt.show()
	kpl, desc = sift.detectAndCompute(img, None)
	#siftOut = open(siftFeatures + filename[:-4], "w+")
	# imgShow = cv2.drawKeypoints(imgGray, kpl, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# plt.imshow(imgShow)
	# plt.show()

	# pdb.set_trace()
	# break;
	infile = open('codebook','rb')
	codebook = pickle.load(infile)

	labels = codebook.predict(desc)
	labels = labels.reshape((1, 2000))

	query_histogram = np.zeros((1, 1000))
	for label in labels[0]:
		query_histogram[0][label] += 1
	# pdb.set_trace()

	histograms = np.load('histograms.npy')

	distances = []

	for idx, histogram in enumerate(histograms):
		distance = np.linalg.norm(histogram - query_histogram[0])
		distances.append((idx, distance))

	# pdb.set_trace()

	distances = sorted(distances, key=lambda x: x[1])

	# pdb.set_trace()

	images = os.listdir(directory)
	images.sort()

	pdb.set_trace()

	outfile = open(results + 'all_souls_ranked_1.txt', 'w')
	for i in range(10):
		idx = distances[i][0]
		img_name = images[idx]
		outfile.write(img_name + '\n')

	outfile.close()


if __name__ == "__main__":
    main()