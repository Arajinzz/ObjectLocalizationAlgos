import cv2
import os
import numpy as np
import keras
import pickle


mymodel = keras.models.load_model('../models/sift_model.h5')
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



def determineClass(predictions):
    return np.argmax(predictions[0])


def validate(testpath):
    c = -1
    total = 0
    positive = 0
    for d1 in os.listdir(testpath):

        p1 = testpath + '/' + d1
        
        c += 1

        #if d1 == 'bike' or d1 == 'car' or d1 == 'human' or d1 == 'human' or d1 == 'noise' or d1 == 'boat' or d1 == 'canoe':
        #    continue

        class_percentage = 0
        class_total = 0
        class_negative = 0

        for img_name in os.listdir(p1):

            img_path = p1 + '/' + img_name
            image = cv2.imread(img_path, 0)
            
            descs = getDescriptor(image).reshape(1, k)
            prediction = mymodel.predict(descs)

            cc = determineClass(prediction)

            if cc == c:
                #print(img_path, 'Classified POSITIVE')
                class_percentage += 1
                positive += 1
            else:
                class_negative += 1
                #print(getClass(cc))
                #print(img_path, 'Classified Negative as : ', getClass(cc))

            total += 1
            class_total +=1

        class_percentage /= class_total
        print('Class', getClass(c), '\tTotal Data : ', class_total,'\tNegative : ', class_negative, '\tAccuracy : ', class_percentage)

    return positive / total




accuracy = validate('../testing_data')

print(accuracy)


#descriptors = getDescriptor(input_img)
#prediction = mymodel.predict(descriptors)