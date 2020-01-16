# -*- coding: utf-8 -*-


# LeNet for MNIST using Keras and TensorFlow
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, Reshape
# downloading and determine dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1,28,28,1) / 255.0, x_test.reshape(-1,28,28,1) / 255.0

# create model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',input_shape=(28,28,1)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(10, activation='softmax'))




model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
## normal train
#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test,  y_test, verbose=2)

# batch selecters
#def random_batch(X,y, size=1):
    #length = len(X)
    #indices = np.random.choice(length,size)
    #return X[indices],y[indices]

def order_batch(X, y, itr, size=10):
    return X[itr*size:(itr+1)*size],y[itr*size:(itr+1)*size]

batch_size = 300
epoch = 2

#%% first train for initial loss and accuracy values
#X_batch, y_batch = random_batch(x_train,y_train,batch_size)
#loss, acc = model.train_on_batch(X_batch, y_batch)
#print("epoch : none , loss : %f , acc : %f" % (loss,acc))

#%% lets train all
for e in range(epoch):
    for b in range(int(len(x_train)/batch_size)):
       #X_batch, y_batch = random_batch(x_train,y_train,batch_size)
        # or
        X_batch, y_batch = order_batch(x_train,y_train,b,batch_size)
        loss, acc = model.train_on_batch(X_batch, y_batch)
        model.save_weights("model%02d%04d.h5" % (e+1,b+1))
    print("epoch : %04d , loss : %f , acc : %f" % (e+1,loss,acc))
    
    
#a = conv2d_6000['conv2d_6']['kernel:0'][0,0,0,0]

