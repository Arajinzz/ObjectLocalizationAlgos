import os
import shutil

input_folder = 'train_data_binary'

for d1 in os.listdir(input_folder):
    p1 = input_folder + '/' + d1
    
    for d2 in os.listdir(p1):
        
        p2 = p1 + '/' + d2

        for d3 in os.listdir(p2):

            p3 = p2 + '/' + d3
            
            for img_name in os.listdir(p3):

                img_path = p3 + '/' + img_name
                savein = p1 + '/' + p3.split('/')[2] + '_' + p3.split('/')[3] + '_' + img_name

                shutil.copyfile(img_path, savein)
                



        




