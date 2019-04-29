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


images = 'oxbuild_images/'

def main():
	codebook = np.load('codebook.npy')
	# 6. Produce the labels for each of the test samples
    # labels = codebook.predict(X_test)
    # labels2 = codebook.predict(X_train)

	histogram = np.zeros((5063, 5000))
	print(len(os.listdir(images)))
	pdb.set_trace()

    # print(X_train.shape)

    # # 7. Reshape to appropriate sizes 
    # X_train = X_train.reshape(1500, 16, 768)
    # X_test = X_test.reshape(500, 16, 768)

    # # 8. TODO Build your histogram
    # labels2 = np.reshape(labels2, (1500, 16))
    # X_hist_train = np.zeros((1500, 15))
    # for i, label in enumerate(labels2):
    #     for j in label:
    #         for k in range(15):
    #             if j == k:
    #                 X_hist_train[i][k] += 1

    # labels = np.reshape(labels, (500, 16))
    # X_hist_test = np.zeros((500, 15))
    # for i, label in enumerate(labels):
    #     for j in label:
    #         for k in range(15):
    #             if j == k:
    #                 X_hist_test[i][k] += 1
    
    # # 9. Train the classifier
    # train_classifier(X_hist_train, X_hist_test, y_train, y_test)	

if __name__ == "__main__":
    main()
