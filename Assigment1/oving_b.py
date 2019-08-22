
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 16:02:04 2019

@author: anette


OPPGAVE 1 B:
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('day_length_weight.csv')

print (data.describe())


numpy_data = data.values

y = np.mat([[y] for y in data.iloc[:, 0].values])
x = np.mat(numpy_data[:, 1:3])




x_train = x[0:500, :]
y_train = y[0:500]

x_test = x[500:, :]
y_test = y[500:]

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([[0.0]])


        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(100000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_test, model.y: y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


x1_ends = np.mat([[np.min(x[:,0])], [np.max(x[:, 0])]])
x2_ends = np.mat([[np.min(x[:,1])], [np.max(x[:, 1])]])
#y_ends = np.mat( [[W[0,0]*x_ends[0,0] + b[0,0]], [W[0,0]*x_ends[1,0] + b[0,0]]])


#plt.plot(x_ends, y_ends, c='red')
ax.scatter(x[:,0], x[:,1], y, label='Input data', alpha=0.25)


plt.legend()
plt.show()
