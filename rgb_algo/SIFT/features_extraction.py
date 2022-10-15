import numpy as np
import os
import cv2
import pickle

from sklearn.cluster import MiniBatchKMeans

input_images_path = '../training_data'

detector = cv2.xfeatures2d.SIFT_create(3000)

dictionary = []
images = []
labels = []

print('Getting Descriptors')


# GET SIFT DESCRIPTORS
for path in os.listdir(input_images_path):
    label = path
    path = input_images_path + '/' + path
    for img_name in os.listdir(path):
        img_path = path + '/' + img_name
        img = cv2.imread(img_path, 0)
        images.append(img)
        labels.append(label)
        kp, des = detector.detectAndCompute(img, None)

        if not des is None:
            for d in des:
                dictionary.append(d)


print(labels)


# N CLASSES * 10
k = 9 * 1500
batch_size = len(images) * 3

print('KMEANS Training')
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(dictionary)


print('Creating Histograms')


# Histogram Creation
kmeans.verbose = False

hist_list = []


for img in images:
    kp, des = detector.detectAndCompute(img, None)
    hist = np.zeros(k)

    if not des is None:
        nkp = np.size(kp)

        idx = kmeans.predict(des)

        # Normalized Histogram
        np.add.at(hist, idx, 1/nkp)

    hist_list.append(hist)




def getVectorLabel(name):
    
    if name == 'bike':
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])

    if name == 'boat':
        return np.array([0, 1, 0, 0, 0, 0, 0, 0, 0])

    if name == 'canoe':
        return np.array([0, 0, 1, 0, 0, 0, 0, 0, 0])

    if name == 'car':
        return np.array([0, 0, 0, 1, 0, 0, 0, 0, 0])

    if name == 'human':
        return np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])

    if name == 'noise':
        return np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
    
    if name == 'pickup':
        return np.array([0, 0, 0, 0, 0, 0, 1, 0, 0])

    if name == 'truck':
        return np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])

    if name == 'van':
        return np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])

    return np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])



dataset = []

for i in range(len(hist_list)):
    x = hist_list[i]
    y = getVectorLabel(labels[i])

    dataset.append(np.r_[x, y])



print('Saving Dataset and KMEANS model')

dataset = np.array(dataset)
np.save('features.npy', dataset)

with open('kmeans_model.sav', 'wb') as f:
    pickle.dump(kmeans, f)
