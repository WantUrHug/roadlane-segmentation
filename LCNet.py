import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, max_pooling2d, batch_normalization, dropout, flatten, dense
#tensorflow.nn.conv2d是没有附带 relu的,而 tensorflow.layers.conv2d是默认使用 relu 作为激活函数的
#conv2d(input, fitler, kernel_size, strides, padding, dilation)
import numpy as np
import os

class Block():

	def __init__(self, noutput, training):

		self.noutput = noutput
		self.training = training

	def __call__(self, input):

		return self.inference(input)

	def inference(self, input):

		output = conv2d(input, self.noutput, kernel_size = (3, 3), strides = 1, padding = "SAME", activation = None)
		output = batch_normalization(output, self.training)
		output = tf.nn.relu(output)

		output = conv2d(input, self.noutput, kernel_size = (3, 3), strides = 1, padding = "SAME", activation = None)
		output = batch_normalization(output, self.training)
		output = tf.nn.relu(output)

		output = max_pooling2d(output, pool_size = (2, 2), strides = (2, 2))

		return output

class Net():

	def __init__(self, output_channels, training):

		self.output_channels = output_channels
		#self.linw = int(w/8)
		#self.linh = int(h/8)
		self.training = training

		self.layers = []
		self.layers.append(Block(16, training))
		self.layers.append(Block(32, training))
		self.layers.append(Block(64, training))

	def __call__(self, input):

		return self.inference(input)

	def inference(self, input):

		output = input
		for layer in self.layers:
			output = layer(output)

		output = flatten(output)
		output = dense(output, 1024)
		output = batch_normalization(output, self.training)
		output = tf.nn.relu(output)

		output = dense(output, self.output_channels)

		return output

if __name__ == "__main__":

	a = tf.placeholder(tf.float32, [1, 1024, 512, 1])
	net = Net(8, True)
	print(net(a))