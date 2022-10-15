import cv2
import os
import numpy as np
import keras
import pickle


mymodel = keras.models.load_model('../models/sift_model.h5.h5')
kmeans = pickle.load(open('../models/kmeans_model.sav', 'rb'))
detector = cv2.xfeatures2d.SIFT_create(3000)
k = 9 * 1500

def getDescriptor(input_img):
    kp, des = detector.detectAndCompute(input_img, None)
    hist = np.zeros(k)
    if not des is None:
        nkp = np.size(kp)

        idx = kmeans.predict(des)

        # Normalized Histogram
        np.add.at(hist, idx, 1/nkp)
    
    return hist


def getClass(c):
    if c == 0:
        return "bike"
    
    if c == 1:
        return "boat"

    if c == 2:
        return "canoe"

    if c == 3:
        return "car"
    
    if c == 4:
        return "human"
    
    if c == 5:
        return "noise"
    
    if c == 6:
        return "pickup"

    if c == 7:
        return "truck"
    
    if c == 8:
        return "van"



input_img = '../testing_data/car/blizzard_1_52.png'
input_img = cv2.imread(input_img, 0)

descriptors = getDescriptor(input_img).reshape(1, k)
prediction = mymodel.predict(descriptors)
index = np.argmax(prediction)
cl = getClass(index)

print('Predicted as : ', cl)
cv2.imshow('Image', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()