
# -*- coding: utf-8 -*-
"""
Lag en modell som predikerer tilsvarende NAND-operatoren.
Visualiser resultatet etter optimalisering av modellen.

2 dimensjonal data, trenger 3d visualisering
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Qt5Agg')


x_train = np.mat([ [0, 0], [0, 1], [1, 0], [1, 1], [0.4, 1], [1, 0.4], [0.6, 1], [1, 0.6]  ])
y_train = np.mat([ [1],    [1],    [1],    [0],     [1],      [1],      [0],      [0]       ])


x_test = np.mat([ [0, 0], [0, 1], [1, 0], [1,1]  ])
y_test = np.mat([ [1],    [1],    [1],    [0]    ])


class SigmoidRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0], [0.0]])
        self.b = tf.Variable([[0.0]])

        # Logits
        logits = tf.matmul(self.x, self.W) + self.b
        
        #Predictor
        f = tf.sigmoid(logits)

        # Mean Squared Error
        self.loss = tf.losses.sigmoid_cross_entropy(self.y, logits)


model = SigmoidRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(100).minimize(model.loss)

session = tf.Session()


# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

start = time.time()

print('global variables initialized')

for epoch in range(3000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if epoch%10000 == 0:
        print(epoch)

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_test, model.y: y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

end = time.time()

print ( end - start)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x_plot = np.linspace(0,1)
z_plot = np.linspace(0,1)

X, Z = np.meshgrid(x_plot, z_plot)

y_real = np.mat(  1/(1 +np.e**-(X * W[0, 0] + Z * W[1, 0]  + b[0, 0]) )    ).transpose()

ax.plot_surface(X, Z, y_real, label ='prediction')

ax.scatter(X, Z, y_real)




plt.legend()
plt.show()