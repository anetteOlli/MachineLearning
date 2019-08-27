
# -*- coding: utf-8 -*-

"""
Lag en modell som predikerer tilsvarende NOT-operatoren.
Visualiser resultatet etter optimalisering av modellen.

Har endimensjonale input. Trenger kun 2D visualisering
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time



#floats are added to make the model able to intrerper 
x_train = np.mat([[0], [1], [0.3], [0.7], [0],  [0.5], [0.4]  ])
y_train = np.mat([[1], [0], [1],   [0],   [1],  [0],   [1]     ])

x_test = np.mat([ [0], [0], [0], [0], [1], [1], [1], [1] ])
y_test = np.mat([ [1], [1], [1], [1], [0], [0], [0], [0] ])

class SigmoidRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
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

for epoch in range(30000):
    session.run(minimize_operation, {model.x: x_train, model.y: y_train})
    if epoch%10000 == 0:
        print(epoch)

# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], {model.x: x_test, model.y: y_test})
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

end = time.time()

print ( end - start)


#Generate graph
x_ends = np.arange(0.0, 1.05, 0.005)

y_ends = np.mat(  1/(1 +np.e**-(x_ends * W[0, 0] + b[0, 0]) )    ).transpose()

plt.plot(x_train, y_train, 'o', label='Input data')
plt.plot(x_ends, y_ends, c='red')

plt.legend()
plt.show()

