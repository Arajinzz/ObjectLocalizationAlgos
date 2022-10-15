import cv2
import os
import numpy as np
import keras
import pickle


mymodel = keras.models.load_model('../models/hog_model.h5')
winSize = (20,20)
blockSize = (10,10)
blockStride = (5,5)
cellSize = (10,10)
nbins = 9
derivAperture = 1
winSigma = -1.
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
useSignedGradients = True
 
handler = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, useSignedGradients)


def getDescriptor(input_img):
    patchsize = 64
    strides = 16
    h, w = input_img.shape
    chunks = []

    if h >= patchsize and w < patchsize:
        input_img = cv2.resize(input_img, (patchsize, h))
        w = patchsize
    elif h < patchsize and w >= patchsize:
        input_img = cv2.resize(input_img, (w, patchsize))
        h = patchsize

    if (h > patchsize and w >= patchsize) or (h >= patchsize and w > patchsize):
        for i in range(0, h-patchsize, strides):
            for j in range(0, w-patchsize, strides):
                block = input_img[i:i+patchsize, j:j+patchsize]

                block = cv2.resize(block, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
                chunks.append(block)
        
        lastline = h-patchsize
        for j in range(0, w-patchsize, strides):
            block = input_img[lastline:lastline+patchsize, j:j+patchsize]

            block = cv2.resize(block, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
            chunks.append(block)

        lastcolumn = w-patchsize

        for i in range(0, h-patchsize, strides):
            block = input_img[i:i+patchsize, lastcolumn:lastcolumn+patchsize]
            block = cv2.resize(block, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
            chunks.append(block)

        block = input_img[h-patchsize:h, w-patchsize:w]
        if(block.sum() >= 120 * 20):
            block = cv2.resize(block, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
            chunks.append(block)
    else:
        input_img = cv2.resize(input_img, (patchsize, patchsize), interpolation=cv2.INTER_CUBIC)
        chunks.append(input_img)


    to_predict = []

    for chunk in chunks:
        to_predict.append(handler.compute(chunk).flatten())


    to_predict = np.array(to_predict)

    return to_predict


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



def determineClass(predictions):
    dictionary = {'0':[0, 0], '1':[0, 0], '2':[0, 0], '3':[0, 0], '4':[0, 0], '5':[0, 0]
                  ,'6':[0, 0], '7':[0, 0], '8':[0, 0], '9':[0, 0]}
    for p in predictions:
        c = np.argmax(p)
        dictionary[str(c)][0] += 1
        dictionary[str(c)][1] += p[c]

    arr = []
    for key in dictionary:
        arr.append([int(key), dictionary[key][0], dictionary[key][1]])

    arr = sorted(arr, key=lambda s: s[1])
    return arr[-1][0]




input_img = '../testing_data/car/blizzard_1_52.png'
input_img = cv2.imread(input_img, 0)

descriptors = getDescriptor(input_img)
prediction = mymodel.predict(descriptors)
index = determineClass(prediction)
cl = getClass(index)

print('Predicted as : ', cl)
cv2.imshow('Image', input_img)
cv2.waitKey(0)
cv2.destroyAllWindows()