
# -*- coding: utf-8 -*-
"""
Lag en modell med prediktoren f (x) = softmax(xW + b) som
klassifiserer handskrevne tall. Se mnist for eksempel lasting av
MNIST datasettet, og visning og lagring av en observasjon. Du
skal oppnå en nøyaktighet på 0.9 eller over. Lag 10 .png bilder
som viser W etter optimalisering.

link til MNIST: https://gitlab.com/ntnu-tdat3025/ann/mnist
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
#from sklearn.datasets import fetch_openml
#from sklearn.model_selection import train_test_split as tts
import random as rnd
matplotlib.use('Qt5Agg')


# Loading dataset and splitting to training and test set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784)
y_train = y_train.reshape(60000,1)
x_test = x_test.reshape(10000, 784)
y_test = y_test.reshape(10000,1)


# Fixing ys
real_y_train = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(len(y_train))])
for i in range(len(y_train)):
    real_y_train[i, int(y_train[i, 0])] = 1
real_y_test = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(len(y_test))])
for i in range(len(y_test)):
    real_y_test[i, int(y_test[i, 0])] = 1

class SoftmaxModel:
    def __init__(self):
        self.x = tf.placeholder("float", [None, 784])
        self.y = tf.placeholder(tf.float32)

        # Model variables
        #self.W = tf.Variable(tf.zeros([784,10]))
        self.b = tf.Variable(tf.random.truncated_normal([10]))
        
        self.W = tf.Variable(tf.random.truncated_normal([784, 10]))
      

    
        # Logits
        logits = tf.matmul(self.x, self.W) + self.b
        
        #First layer function
        self.f = tf.nn.softmax(logits, name=None)
        
        
        self.loss = tf.losses.softmax_cross_entropy(self.y, logits)

model = SoftmaxModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(1000).minimize(model.loss)

session = tf.Session()


# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

start = time.time()

print('global variables initialized')

for epoch in range(2000):
    session.run(minimize_operation, {model.x: x_train, model.y: real_y_train})
    if epoch%50 == 0:
        print(epoch)

# Evaluate training accuracy
W, b, loss, preds = session.run([model.W, model.b, model.loss, model.f], {model.x: x_test, model.y: real_y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

for i in range(10):
    plt.imsave((str(i) + '.png'), W[:,i].reshape(28,28) )
    



session.close()

print(sum([int(list(preds[i]) == list(real_y_test[i])) for i in range(len(preds))])/len(preds)*100)

