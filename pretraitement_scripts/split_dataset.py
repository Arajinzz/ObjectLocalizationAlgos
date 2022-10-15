import os
import random
import shutil

# replace this with the folder you wanna split
input_folder = 'data_rgb'

train_save = 'training_data'
test_save = 'testing_data'

test_bike_size = 0.1
test_boat_size = 0.4
test_canoe_size = 0.3
test_car_size = 0.4
test_human_size = 0.4
test_noise_size = 0.2
test_pickup_size = 0.2
test_truck_size = 0.3
test_van_size = 0.3


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
    bike_images.append(bike + '/' + di)


for di in os.listdir(boat):
    boat_images.append(boat + '/' + di)


for di in os.listdir(canoe):
    canoe_images.append(canoe + '/' + di)


for di in os.listdir(car):
    car_images.append(car + '/' + di)


for di in os.listdir(human):
    human_images.append(human + '/' + di)


for di in os.listdir(noise):
    noise_images.append(noise + '/' + di)


for di in os.listdir(pickup):
    pickup_images.append(pickup + '/' + di)


for di in os.listdir(truck):
    truck_images.append(truck + '/' + di)


for di in os.listdir(van):
    van_images.append(van + '/' + di)


# SHUFFLE
random.shuffle(bike_images)
random.shuffle(boat_images)
random.shuffle(canoe_images)
random.shuffle(car_images)
random.shuffle(human_images)
random.shuffle(noise_images)
random.shuffle(pickup_images)
random.shuffle(truck_images)
random.shuffle(van_images)



def splittraintest(images, size):
    train_images = images[:1-int(len(images)*size)]
    test_images = images[1-int(len(images)*size):]

    return train_images, test_images


# SPLIT
train_bike_images, test_bike_images = splittraintest(bike_images, test_bike_size)
train_boat_images, test_boat_images = splittraintest(boat_images, test_boat_size)
train_canoe_images, test_canoe_images = splittraintest(canoe_images, test_canoe_size)
train_car_images, test_car_images = splittraintest(car_images, test_car_size)
train_human_images, test_human_images = splittraintest(human_images, test_human_size)
train_noise_images, test_noise_images = splittraintest(noise_images, test_noise_size)
train_pickup_images, test_pickup_images = splittraintest(pickup_images, test_pickup_size)
train_truck_images, test_truck_images = splittraintest(truck_images, test_truck_size)
train_van_images, test_van_images = splittraintest(van_images, test_van_size)

print('SAVING IMAGES')


def crFolder(path):
    if(not os.path.exists(path)):
        os.mkdir(path)


crFolder(train_save)
crFolder(test_save)


crFolder(train_save+'/bike')
crFolder(train_save+'/boat')
crFolder(train_save+'/canoe')
crFolder(train_save+'/car')
crFolder(train_save+'/human')
crFolder(train_save+'/noise')
crFolder(train_save+'/pickup')
crFolder(train_save+'/truck')
crFolder(train_save+'/van')


crFolder(test_save+'/bike')
crFolder(test_save+'/boat')
crFolder(test_save+'/canoe')
crFolder(test_save+'/car')
crFolder(test_save+'/human')
crFolder(test_save+'/noise')
crFolder(test_save+'/pickup')
crFolder(test_save+'/truck')
crFolder(test_save+'/van')


def saveImages(images, savepath):
    for img_path in images:
        img_name = img_path.split('/')[-1]
        shutil.copy(img_path, savepath+'/'+img_name)


saveImages(train_bike_images, train_save+'/bike')
saveImages(train_boat_images, train_save+'/boat')
saveImages(train_canoe_images, train_save+'/canoe')
saveImages(train_car_images, train_save+'/car')
saveImages(train_human_images, train_save+'/human')
saveImages(train_noise_images, train_save+'/noise')
saveImages(train_truck_images, train_save+'/truck')
saveImages(train_pickup_images, train_save+'/pickup')
saveImages(train_van_images, train_save+'/van')


saveImages(test_bike_images, test_save+'/bike')
saveImages(test_boat_images, test_save+'/boat')
saveImages(test_canoe_images, test_save+'/canoe')
saveImages(test_car_images, test_save+'/car')
saveImages(test_human_images, test_save+'/human')
saveImages(test_noise_images, test_save+'/noise')
saveImages(test_truck_images, test_save+'/truck')
saveImages(test_pickup_images, test_save+'/pickup')
saveImages(test_van_images, test_save+'/van')
