import warnings
warnings.filterwarnings(action='ignore')

# Importing all required libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,MaxPool2D,Conv2D,Flatten,Dropout
from keras.utils import to_categorical
from PIL import Image
import os
from sklearn.metrics import accuracy_score

print('imports have been completed.')
# Reading image data and storing it into data and labels lists in the form of an array.

data=[]
labels=[]
classes=43
cur_path = os.getcwd()

print('Loading DATA...')

for i in range(classes):
	path = os.path.join(cur_path,'Data/Train',str(i))
	images = os.listdir(path)

	for img in images:
		try:
			image_i = Image.open(path+'\\'+img)
			image_i = image_i.resize((30,30))
			image_i = np.array(image_i)
			data.append(image_i)
			labels.append(i)
		except:
			print('ERROR WHILE LOADING IMAGES...')

print('DATA has been loaded.')

data = np.array(data)
labels = np.array(labels)

# spliting the data into train and test sets.
print('Splitting the data...')

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,random_state=101)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print('Splitting is done.')

# Defining CNN Model by instantiating Sequential model present in keras to introduce all sequnetial properties in our model.
print('Making model...')

model = Sequential()

# Defining first layer of CNN.

model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

# Defining Secong layer of CNN.

model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

# Adding a Flatten layer to CNN model to squeeze the layers to one dimension.

model.add(Flatten())

# At last adding a fully connected layer to CNN.

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(43, activation='softmax'))

print('Model building has been completed.')

# Compiling CNN.
print('Starting model compilation...')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('Model Compilation has been completed.')

# Training the CNN model.
print('Training Model it will take a while...')

epochs = 20
model_info = model.fit(x_train, y_train, batch_size=64, epochs = epochs, validation_data=(x_test, y_test))

print('Training has been completed.')

# Saving the Model.
print('Saving the Model...')

model.save('Traffic_Sign_Recognition_model.h5')

print('Model has been saved.')

# Testing the model.

test = pd.read_csv('Data/Test.csv')
labels = test.ClassId
images = 'Data/' + test.Path


print('Testing the model...')

test_data = []

for img in images:
	image = Image.open(img)
	image = image.resize((30,30))
	test_data.append(np.array(image))

test_input = np.array(test_data)
pred = model.predict_classes(test_input)

print(f'accuracy score of model is : {accuracy_score(pred, labels)}')