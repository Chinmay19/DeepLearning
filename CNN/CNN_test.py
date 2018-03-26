import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # no warning 
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist  = input_data.read_data_sets("/media/chinmay/Important/ABHYAS/playground/tensorflow/data", one_hot = True )

n_classes = 10
batch_size = 128

#height x width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder(tf.int64)

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2D(x, W):
	return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool2D(x):
	return tf.nn.max_pool(x, ksize = [1,2,2,1], strides =[1,2,2,1], padding = 'SAME')




def conv_net_model(data):

	# (input_data * weights) + biases

	weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
			   'out':tf.Variable(tf.random_normal([1024, n_classes]))}

	biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
			   'b_conv2':tf.Variable(tf.random_normal([64])),
			   'b_fc':tf.Variable(tf.random_normal([1024])),
			   'out':tf.Variable(tf.random_normal([n_classes]))}

	x_rs = tf.reshape(x, shape = [-1, 28, 28, 1])

	conv1 = conv2D(x_rs,weights['W_conv1'])
	conv1 = max_pool2D(conv1)

	conv2 = conv2D(conv1, weights['W_conv2'])
	conv2 = max_pool2D(conv2)

	fc = tf.reshape(conv2,[-1,7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out'])+ biases['out']



	
	return output

def train_network(x):
	prediction = conv_net_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))

	# learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + back prop
	num_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(num_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				iter_x, iter_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict = {x: iter_x, y: iter_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', num_epochs, 'loss:',epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_network(x)
