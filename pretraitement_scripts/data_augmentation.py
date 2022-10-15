import cv2
import numpy as np
import os

# FOLDERS TO DO DATA AUGMENTATION ON THEM
# VAN       64 image
# TRUCK    149 image
# PICKUP    67 image
# NOISE     68 image
# BIKE      49 image

augment_folders = ['van', 'truck', 'pickup', 'noise', 'bike']

input_folder = 'data_rgb'

for d in os.listdir(input_folder):
    if d in augment_folders:
        p = input_folder + '/' + d
        for dd in os.listdir(p):
            img_path = p + '/' + dd
            img = cv2.imread(img_path)
            img = cv2.flip(img, 1)
            cv2.imwrite(p+'/'+dd.split('.')[0]+'_flipped.png',img)
