import csv
from matplotlib import pyplot as plt
import numpy as np
#Read in data locations from csv file
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#Create and fill X_train and y_train using training data
images = []
steering_angle = []
#Correction for using center left and right images respectively
correction = [0,0.15,-0.15]
for line in lines:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = './data/IMG/'+filename
        image = plt.imread(current_path)
        images.append(image)
        steering_angle.append(float(line[3])+float(correction[i]))

X_train = np.array(images)
y_train = np.array(steering_angle)

#Model Architecture
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Cropping2D, Convolution2D

model = Sequential()
#Preprocessing
model.add(Lambda(lambda x:(x/255.0)-0.5,input_shape=( 160, 320, 3)))
model.add(Cropping2D(cropping = ((70,25),(0,0))))
#Convolutional Layers
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
#Dense Layers
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Compiling and Saving the model
model.compile(loss='mse',optimizer='adam')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=3)
model.save('model.h5')
