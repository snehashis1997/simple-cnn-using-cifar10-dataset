# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 01:03:21 2019

@author: user
"""

import numpy as np
import cv2
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense,Flatten
from keras.utils.np_utils import to_categorical
from sklearn.metrics import classification_report

import numpy as np

name='my_cnn.hdf5'
model=Sequential()
model=Sequential()
model.add(Convolution2D(32,4,4,input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Convolution2D(32,4,4,input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
 
model.add(Flatten())
 
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.load_weights(name)
frame=cv2.imread('cat.jpg')

roi=frame
x1=20
y1=30
x2=600
y2=400
lables=['airplane','automob','bird','cat','deer','dog','frog','horse','ship','truck']
font  = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 3
fontColor              = (0,0,255)
lineType  =2      
roi=roi/255
roi=cv2.resize(roi,(32,32))
roi=np.expand_dims(roi,axis=0)
prob=model.predict(roi)
clas=model.predict_classes(roi)
clas=int(clas)
prob=prob.reshape(10,1)
   # if (prob[])>.6:
text = lables[clas] +str(max(prob))
cv2.putText(frame, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    #prediction=model(frame)
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
cv2.imshow('frame',frame)
# When everything done, release the capture
cv2.waitKey(0)
cv2.destroyAllWindows()


