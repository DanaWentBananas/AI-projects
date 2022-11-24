import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as n

(xTrain,yTrain),(xTest,yTest) = keras.datasets.mnist.load_data()

#info
print(f'Number of taining samples: {xTrain.shape[0]}')
print(f'Number of test samples: {xTest.shape[0]}')
print(f'Shape of individual img sample {xTrain[0].shape}')

#Example of an img of a digit
#plt.matshow(xTrain[0])
#plt.show()

#Scale images
xTrain = xTrain/255
xTest = xTest/255

#Flatten images
# print(f'Before flattening: {xTrain.shape}')
# xTrainflat = xTrain.reshape(len(xTrain),28*28)
# print(f'After flattening: {xTrainflat.shape}')
# xTestflat = xTest.reshape(len(xTest),28*28)


#make model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(10, activation = 'sigmoid')
])

#compile model
model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#train
model.fit(xTrain, yTrain, epochs=5)

#evaluate accuracy on test dataset
model.evaluate(xTest, yTest)

#predict
predictions = model.predict(xTest)
print(n.argmax(predictions[0]))

#save
if os.path.isfile('stuff/DigitsRecognizer.h5') is False:
    model.save('stuff/DigitsRecognizer.h5')

