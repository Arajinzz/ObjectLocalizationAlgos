import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import tensorflow as tf
import keras


def overlap(l1, r1, l2, r2):

    R1 = set(range(l1[0], r1[0]))
    R2 = set(range(l2[0], r2[0]))
    R3 = set(range(l1[1], r1[1]))
    R4 = set(range(l2[1], r2[1]))

    return R1 & R2 and R3 & R4


def slide(image, windowSize):
    h, w = image.shape
    
    keypoints, descriptors = detector.detectAndCompute(image, None)
    hogs = []
    paddI = (windowSize[0] - 1) // 2
    paddJ = (windowSize[1] - 1) // 2

    for keypoint in keypoints:
        j, i = keypoint.pt
        
        i = int(i)
        j = int(j)

        beginI = int(max(i - 30, 0))
        endI = int(min(i + 30, h-1))

        beginJ = int(max(j - 30, 0))
        endJ = int(min(j + 30, w-1))

        block = image[beginI:endI, beginJ:endJ]
        block = cv2.resize(block, (60, 60), interpolation=cv2.INTER_CUBIC)

        #fd = hog(block, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, multichannel=False)
        fd = handler.compute(block)
        fd = fd.flatten()
        fd = fd.reshape(1, fd.shape[0])
        p = mymodel.predict(fd)

        toRemove = []
        doAdd = True
        ind = 0
        # CHECK RECTANGLE IT IT OVERLAPS WITH ANOTHER RECTANGLE
        while ind < len(hogs):
            # x, y
            l = hogs[ind]
            l1 = (l[2], l[1])
            r1 = (l[4], l[3])

            l2 = (beginJ, beginI)
            r2 = (endJ, endI)

            #s1 = generateSet((beginI, endI), (beginJ, endJ))
            #s2 = generateSet((l[1], l[3]), (l[2], l[4]))

            if(overlap(l2, r2, l1, r1)):
                '''if ( l[0][0][0] > p[0][0] ):
                    doAdd = False
                    break
                else:
                    del hogs[ind]
                    ind -= 1'''
                '''hogs[ind][1] = int((l1[1] + beginI) // 2)
                hogs[ind][2] = int((l1[0] + beginJ) // 2)
                hogs[ind][3] = int((r1[1] + endI) // 2)
                hogs[ind][4] = int((r1[0] + endJ) // 2)
                hogs[ind][0][0][0] = (l[0][0][0] + p[0][0]) / 2
                hogs[ind][0][0][1] = (l[0][0][1] + p[0][1]) / 2'''
                
                if np.argmax(p[0]) == np.argmax(l[0][0]):
                    hogs[ind][1] = min(l1[1], beginI)
                    hogs[ind][2] = min(l1[0], beginJ)
                    hogs[ind][3] = max(r1[1], endI)
                    hogs[ind][4] = max(r1[0], endJ)

                #hogs[ind][0][0][0] = (l[0][0][0] + p[0][0]) / 2
                #hogs[ind][0][0][1] = (l[0][0][1] + p[0][1]) / 2
                doAdd = False

            
            ind += 1

        if doAdd:
            hogs.append([p, beginI, beginJ, endI, endJ])

        #cv2.rectangle(image, (beginJ, beginI), (beginJ+160, beginI+160), (255, 255, 255), thickness=1)

    ci = 0
    cj = 0

    for i in range(len(hogs)):
        pred = hogs[i][0]
        p = pred[0]
        c = np.argmax(p)
        #if c == 0:
        ci = hogs[i][1]
        cj = hogs[i][2]
        cei = hogs[i][3]
        cej = hogs[i][4]
        dra = True
        cl = ''
        if c == 0:
            cl = "CAR"
        elif c == 1:
            cl = "BOAT"
        elif c == 2:
            cl = "PERSON"
        else:
            print("NEGATIVE")
            dra=False
        
        if cei - ci > 60:
            if(dra):
                cv2.rectangle(image, (cj, ci), (cej, cei), (255, 255, 255), thickness=1)
                image = cv2.putText(image, cl, (cj, ci-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                #cv2.rectangle(image, (cj, ci), (cj+windowSize[1], ci+windowSize[0]), (255, 255, 255), thickness=1)
                #print(hogs[i][1], hogs[i][2])

    return image


DATAPATH = 'dataset/segmented/fall5'
VIDEO = []

for di in sorted(os.listdir(DATAPATH), key=lambda s: int(s.split('.')[0])):
    VIDEO.append(cv2.imread(DATAPATH + '/' + di, 0))

# X = 450, 570
# Y = 220, 310
'''frame = VIDEO[100]

mymodel = keras.models.load_model('test_model.h5')

img = slide(frame, (160, 160))

plt.imshow(img, cmap='gray')
plt.show()'''

detector = cv2.xfeatures2d.SURF_create(1000)
mymodel = keras.models.load_model('../models/sliding_window_models.h5')
winSize = (2, 2)
blockSize = (2, 2)
blockStride = (2, 2)
cellSize = (2, 2)
nbins = 9
handler = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)


for frame in VIDEO:
    img = slide(frame, (160, 160))

    cv2.imshow('oo', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()