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

directory = 'sift/'

def main():
	firstFile = os.listdir(directory)[0]
	desc = np.load(directory + firstFile)
	# pdb.set_trace()
	size = 20 if desc.shape[0] > 20 else desc.shape[0] - 1
	# pdb.set_trace()
	desc = desc[np.random.randint(desc.shape[0], size=size), :]
	
	allDesc = desc.tolist()
	# pdb.set_trace()

	for filename in os.listdir(directory)[1:]:
		desc = np.load(directory + filename)
		if desc.size > 3:
				size = 20 if desc.shape[0] > 20 else desc.shape[0] - 1
				desc = desc[np.random.randint(desc.shape[0], size=size), :]
				allDesc += desc.tolist()
		# pdb.set_trace()

	allDesc = np.vstack(allDesc)
	allDesc = allDesc.astype(np.float)
	pdb.set_trace()
	codebook = KMeans(n_clusters=1000, max_iter=2, verbose=1).fit(allDesc)
	np.save("allDesc", allDesc)
	# codebook = {'lol': allDesc}
	# np.save("codebook", codebook)
	with open('codebook', 'wb') as codebook_file:
		pickle.dump(codebook, codebook_file)
	# with open('alldesc')


if __name__ == "__main__":
    main()