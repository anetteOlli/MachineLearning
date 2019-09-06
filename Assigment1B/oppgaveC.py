
# -*- coding: utf-8 -*-
"""
) Lag en modell som predikerer tilsvarende XOR-operatoren. Før
du optimaliserer denne modellen må du initialisere
modellvariablene med tilfeldige tall for eksempel mellom -1 og
1. Visualiser både når optimaliseringen konvergerer og ikke
konvergerer mot en riktig modell.


"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Qt5Agg')


x_train = np.mat([ [0, 0], [0, 1], [1, 0], [1, 1], [0.4, 1], [1, 0.4], [0.6, 1], [1, 0.6]  ])
y_train = np.mat([ [0],    [1],    [1],    [0],     [1],      [1],      [0],      [0]       ])


x_test = np.mat([ [0, 0], [0, 1], [1, 0], [1,1]  ])
y_test = np.mat([ [0],    [1],    [1],    [0]    ])


class SigmoidModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W1 = tf.Variable([[10.0, -10.0], [10.0, -10.0]])
        self.b1 = tf.Variable([[-5.0, 15.0]])
        
        
        self.W2 = tf.Variable([[10.0], [10.0]])
        self.b2 = tf.Variable([[-15.0]])

        # Logits
        logits = tf.matmul(self.x, self.W1) + self.b1
        
        #First layer function
        self.f1 = tf.sigmoid(logits)
        
        #Second layer function
        self.f2 = tf.matmul(self.f1, self.W2) + self.b2
        
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.f2)
        
        
model = SigmoidModel()


"""
        logits1 = tf.matmul(self.x, self.W1) + self.b1
        self.h = tf.sigmoid(logits1)
        
        logits2 = tf.matmul(self.h, self.W2) + self.b2
        self.f = tf.sigmoid(logits2)
        
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits2)
"""


# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.000001).minimize(model.loss)

session = tf.Session()


# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

start = time.time()

print('global variables initialized')

for epoch in range(10000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if epoch%10000 == 0:
        print(epoch)

# Evaluate training accuracy
W1, W2, b2, b1, loss = session.run([model.W1, model.W2, model.b2, model.b1, model.loss], {model.x: x_test, model.y: y_test})
print("W1 = %s, b1 = %s, W2= %s, b2 = %s loss = %s" % (W1, b1, W2, b2, loss))

session.close()

end = time.time()

print ( end - start)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x_plot = np.linspace(0,1)
z_plot = np.linspace(0,1)

X, Z = np.meshgrid(x_plot, z_plot)


matrix_hell = ([  (X*W1[0,0] + Z*W1[1,0]), (X*W1[0,1] + Z*W1[1,1] )  ])

x_matrix_hell_1 = matrix_hell[0] + b1[0,0]
x_matrix_hell_2 = matrix_hell[1] + b1[0,1]

f1_matrix_1 = 1 / (1 + np.e**-(x_matrix_hell_1))
f1_matrix_2 = 1/ (1 + np.e**-(x_matrix_hell_2))

y_real = 1 / (1 + np.e**-( f1_matrix_1 * W2[0,0] +f1_matrix_2 * W2[1,0] + b2[0]    ))


ax.plot_surface(X, Z, y_real)
ax.scatter(x_test[:,0], x_test[:,1], y_test, c ='red')




plt.legend()
plt.show()



