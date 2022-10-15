import cv2
import os
import numpy as np


'''winSize = (2, 2)
blockSize = (2, 2)
blockStride = (2, 2)
cellSize = (2, 2)
nbins = 9
handler = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)'''


#FROM : https://www.learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/

# REPLACE THIS WITH HU MOMENTS
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


input_folder = '../training_data'


bike = input_folder + '/bike'
boat = input_folder + '/boat'
canoe = input_folder + '/canoe'
car = input_folder + '/car'
human = input_folder + '/human'
noise = input_folder + '/noise'
pickup = input_folder + '/pickup'
truck = input_folder + '/truck'
van = input_folder + '/van'


bike_images = []
boat_images = []
canoe_images = []
car_images = []
human_images = []
noise_images = []
pickup_images = []
truck_images = []
van_images = []


for di in os.listdir(bike):
    bike_images.append(cv2.imread(bike + '/' + di, 0))


for di in os.listdir(boat):
    boat_images.append(cv2.imread(boat + '/' + di, 0))


for di in os.listdir(canoe):
    canoe_images.append(cv2.imread(canoe + '/' + di, 0))


for di in os.listdir(car):
    car_images.append(cv2.imread(car + '/' + di, 0))


for di in os.listdir(human):
    human_images.append(cv2.imread(human + '/' + di, 0))


for di in os.listdir(noise):
    noise_images.append(cv2.imread(noise + '/' + di, 0))


for di in os.listdir(pickup):
    pickup_images.append(cv2.imread(pickup + '/' + di, 0))


for di in os.listdir(truck):
    truck_images.append(cv2.imread(truck + '/' + di, 0))


for di in os.listdir(van):
    van_images.append(cv2.imread(van + '/' + di, 0))




def getDescriptor(input_img, label):
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
        to_predict.append(np.r_[handler.compute(chunk).flatten(), label])


    to_predict = np.array(to_predict)

    return to_predict


features = []

for frame in bike_images:
    fd = getDescriptor(frame, [1, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in boat_images:
    fd = getDescriptor(frame, [0, 1, 0, 0, 0, 0, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in canoe_images:
    fd = getDescriptor(frame, [0, 0, 1, 0, 0, 0, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in car_images:
    fd = getDescriptor(frame, [0, 0, 0, 1, 0, 0, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in human_images:
    fd = getDescriptor(frame, [0, 0, 0, 0, 1, 0, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in noise_images:
    fd = getDescriptor(frame, [0, 0, 0, 0, 0, 1, 0, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in pickup_images:
    fd = getDescriptor(frame, [0, 0, 0, 0, 0, 0, 1, 0, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in truck_images:
    fd = getDescriptor(frame, [0, 0, 0, 0, 0, 0, 0, 1, 0])
    for i in range(fd.shape[0]):
        features.append(fd[i])


for frame in van_images:
    fd = getDescriptor(frame, [0, 0, 0, 0, 0, 0, 0, 0, 1])
    for i in range(fd.shape[0]):
        features.append(fd[i])


features = np.array(features)
print(features.shape)
np.save('features.npy', features)
