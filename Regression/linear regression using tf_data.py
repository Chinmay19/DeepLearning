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



data, n_samples = utils.read_birth_life_data(DATA_FILE)

dataset = tf.data.Dataset.from_tensor_slices((data[:,0], data[:,1]))
Iterator = dataset.make_initializable_iterator()
X, Y = Iterator.get_next()

plot_loss = []

w = tf.get_variable('weights', initializer = tf.constant(0.0))
b = tf.get_variable('biases', initializer = tf.constant(0.0))

Y_predicted = ((w*X) + b)
loss = tf.square(Y - Y_predicted, name='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)
start = time.time()
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	writer = tf.summary.FileWriter("/media/chinmay/Important/ABHYAS/playground/tensorflow/Week 4/summary", sess.graph)
	
	for i in range(100):
		sess.run(Iterator.initializer)
		total_loss = 0
		try:
			while True:
				_, loss_out = sess.run([optimizer, loss])
				plot_loss.append(loss_out)
				total_loss += loss_out
		except tf.errors.OutOfRangeError:
			pass 
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	writer.close()
	
	w_out, b_out = sess.run([w,b])

print('Took: %f seconds' %(time.time() - start))

plt.plot(plot_loss[:], label = 'MSE')
plt.legend()
plt.show()
plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()