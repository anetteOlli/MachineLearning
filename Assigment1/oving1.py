# -*- coding: utf-8 -*-
"""
Spyder Editor

OPPGAVE 1 A:

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import time

data = pd.read_csv('length_weight.csv')



x = np.mat([[x] for x in data.iloc[:, 0].values])
y = np.mat([[x] for x in data.iloc[:, 1].values])

x_train = x[0:500]
y_train = y[0:500]

x_test = x[500:]
y_test = y[500:]

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
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

print ('/n started working in session /n')

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

start = time.time()

print('global variables initialized')

for epoch in range(300000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if epoch%10000 == 0:
        print(epoch)

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_test, model.y: y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

end = time.time()

print (end - start)

x_ends = np.mat([[np.min(x)], [np.max(x)]])
y_ends = np.mat( [[W[0,0]*x_ends[0,0] + b[0,0]], [W[0,0]*x_ends[1,0] + b[0,0]]])


plt.plot(x_ends, y_ends, c='red')
plt.plot(x, y, 'o', label='Input data', alpha=0.25)


plt.legend()
plt.show()


