# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:08:28 2019

@author: anette
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

data = pd.read_csv('day_head_circumference.csv')

print (data.describe())


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
        f = 20*1/(1+np.e**-(tf.matmul(self.x, self.W) + self.b)) + 31

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.00000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(2000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if epoch % 10000 == 0 :
        print (epoch)

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_test, model.y: y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()


#Generate graph
x_ends = np.arange(0.0, 1750.0, 0.5)

y_ends = np.mat(  20 /(1 +np.e**-(x_ends * W[0, 0] + b[0, 0]) ) +31    ).transpose()



plt.plot(x_ends, y_ends, c='red')
plt.plot(x, y, 'o', label='Input data', alpha=0.25)


plt.legend()
plt.show()

"""
Kjøring ga:
    
W = [[0.00253016]], b = [[-1.4156317e-05]], loss = 2.9476502
"""