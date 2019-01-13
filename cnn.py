import keras
Using TensorFlow backend.
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 19:17:37 2019

@author: user
"""
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report



(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train/255
x_test=x_test/255

y_cat_test=to_categorical(y_test,10)
y_cat_train=to_categorical(y_train,10)

model=Sequential()
model.add(Convolution2D(32,4,4,input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Convolution2D(32,4,4,input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Flatten())
 
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x_train,y_cat_train,verbose=1,epochs=10)
model.evaluate(x_test,y_cat_test)
 
predictions=model.predict_classes(x_test)

name='my_cnn.hdf5'
model.save_weights(name,overwright=True)

#model.load_weights(name)
