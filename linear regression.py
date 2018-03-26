""" Starter code for simple linear regression example using placeholders
Created by Chip Huyen (huyenn@cs.stanford.edu)
CS20: "TensorFlow for Deep Learning Research"
cs20.stanford.edu
Lecture 03
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import utils

'''
	# Step 1: read in data from the .txt file

	# Step 2: create placeholders for X (birth rate) and Y (life expectancy)
	# Remember both X and Y are scalars with type float

	# Step 3: create weight and bias, initialized to 0.0
	# Make sure to use tf.get_variable

	# Step 4: build model to predict Y
	# e.g. how would you derive at Y_predicted given X, w, and b

	# Step 5: use the square error as the loss function

	# Step 6: using gradient descent with learning rate of 0.001 to minimize loss

	# Step 7: initialize the necessary variables, in this case, w and b

	# Step 8: train the model for 100 epochs

	# Step 9: output the values of w and b
		
'''


DATA_FILE = 'birth_life_2010.txt'

# Step 1
data, n_samples = utils.read_birth_life_data(DATA_FILE)
# Step 2
X = tf.placeholder(tf.float32, shape=None, name='X')
Y = tf.placeholder(tf.float32, shape=None, name='Y')
plot_loss = []
# Step 3
w = tf.get_variable('weights', initializer = tf.constant(0.0))
b = tf.get_variable('biases', initializer = tf.constant(0.0))

# Step 4
Y_predicted = ((w*X) + b)
#tf.add(tf.multiply(w, X), b)

# Step 5
loss = tf.square(Y - Y_predicted, name='loss')
#loss = utils.huber_loss(Y, Y_predicted)

# def huber_loss(labels = Y, predictions = Y_predicted, delta=14.0):
	# residual = tf.abs(labels - predictions)
	# def f1(): return 0.5 * tf.square(residual)
	# def f2(): return delta * residual - 0.5 * tf.square(delta)
	# return tf.cond(residual < delta, f1, f2)

# Step 6
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
start = time.time()
# Create a filewriter to write the model's graph to TensorBoard
# writer = tf.summary.FileWriter("/media/chinmay/Important/ABHYAS/playground/tensorflow/Week 4/summary", sess.graph)

with tf.Session() as sess:
	writer = tf.summary.FileWriter("/media/chinmay/Important/ABHYAS/playground/tensorflow/Week 4/summary", sess.graph)
	# Step 7
	sess.run(tf.global_variables_initializer())
	# Step 8
	for i in range(100):
		total_loss = 0
		for x, y in data:
	# Execute train_op and get the value of loss.
	# Don't forget to feed in data for placeholders
			_, l = sess.run([optimizer, loss], feed_dict = {X:x, Y:y})
			plot_loss.append(l) 
			total_loss += l

		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	writer.close()
	
	# Step 9
	w_out, b_out = sess.run([w,b])

print('Took: %f seconds' %(time.time() - start))

#uncomment the following lines to see the plot 
plt.plot(plot_loss[:], label = 'MSE')#or hube loss
plt.legend()
plt.show()
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()