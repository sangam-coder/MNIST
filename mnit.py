import numpy as np
import mnist #Get data set from
import matplotlib.pyplot as plt #Graph
from keras.models import Sequential #ANN architecture
from keras.layers import Dense #the layer in ANN
from keras.utils import to_categorical

#load the DataSet
train_images= mnist.train_images() #training data images
train_labels= mnist.train_labels() #training data labels
test_images= mnist.test_images()
test_labels= mnist.test_labels()

#Normize the images. Normalize the pixel values from [0,255] tp
#[-0.5,0.5] to make our network easier to train

train_images=(train_images/255)-0.5
test_images=(test_images/255)-0.5

#Flatten the images eaxh 28*28 image into 784 dimensional vextor
#to pass into neural network

train_images=train_images.reshape((-1,784))
test_images=test_images.reshape((-1,784))

#print the shape

print(train_images.shape) #60,000 rows and 784 cols
print(test_images.shape) #10,000 rows and 784 cols

#Build the model
#3 layers , 2 layers with 64 neurouns and the relu funtion
#1 layer with 10 neuron and softmax function

model=Sequential()
model.add(Dense(64,activation='relu',input_dim=784))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='relu'))

#Compile the model
# The loss function measure how well the model did on training,and then tries to improve on it using the optimizer

model.compile(
    optimizer='adam',
      loss='categorical_crossentropy', #(classes that are greater than 2)
      metrics=['accuracy']
)

#train the model

model.fit(
    train_images,
      to_categorical(train_labels),
      epochs=5,
      batch_size=32
)

#Evalute the model
model.evaluate(
    test_images,
     to_categorical(test_labels)
)

predictions=model.predict(test_images[:5])
print(np.argmax(predictions,axis=1))
print(test_labels[:5])

for i in range(0,5):
  first_image=test_images[i]
  first_image=np.array(first_image,dtype='float')
  pixels=first_image.reshape((28,28))
  plt.imshow(pixels,cmap='gray')
  plt.show()