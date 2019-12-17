import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, max_pooling2d, batch_normalization, dropout
#tensorflow.nn.conv2d是没有附带 relu的,而 tensorflow.layers.conv2d是默认使用 relu 作为激活函数的
#conv2d(input, fitler, kernel_size, strides, padding, dilation)
import numpy as np
import os

class DownsamplerBlock():

	def __init__(self, noutput, training):
	
		self.noutput = noutput
		self.training = training
	
	def inference(self, input):

		conv = conv2d(input, self.noutput, kernel_size = (3, 3), strides = (2, 2), padding = "SAME")
		maxpool = max_pooling2d(input, pool_size = (2, 2), strides = 2)
		concat = tf.concat([conv, maxpool], -1)
		bn =  batch_normalization(concat, self.training)

		return tf.nn.relu(bn)

class NonBottleneck_1D():

	def __init__(self, output_channels, dropprob, dilated, training):

		self.output_channels = output_channels
		self.dropprob = dropprob
		self.dilated = dilated
		self.training = training

	def inference(self, input):

		output = conv2d(input, self.output_channels, kernel_size = (3, 1), strides = 1, padding = "SAME")
		output = conv2d(output, self.output_channels, kernel_size = (1, 3), strides = 1, padding = "SAME", activation = None)
		output = batch_normalization(output, self.training)
		output = tf.nn.relu(output)

		output = conv2d(output, self.output_channels, kernel_size = (3, 1), strides = 1, dilation_rate = (self.dilated, self.dilated), padding = "SAME")
		output = conv2d(output, self.output_channels, kernel_size = (1, 3), strides = 1, dilation_rate = (self.dilated, self.dilated), padding = "SAME", activation = None)
		output = batch_normalization(output, self.training)
		if self.dropprob != 0:
			output = dropout(output, rate = self.dropprob, training = self.training)

		return tf.nn.relu(output + input)

class Encoder():

	def __init__(self, num_classes = 1, training = True):

		self.num_classes = num_classes
		#self.training = training
		self.layers = []

		self.initial_block = DownsamplerBlock(13, training)#13=16-3
		#self.layers.append(self.initial_block)
		self.layers.append(DownsamplerBlock(48, training))#48=64-16

		for i in range(5):
			self.layers.append(NonBottleneck_1D(64, 0.03, 1, training))

		self.layers.append(DownsamplerBlock(64, training))#64=128-64
		for i in range(2):
			self.layers.append(NonBottleneck_1D(128, 0.3, 2, training))
			self.layers.append(NonBottleneck_1D(128, 0.3, 4, training))
			self.layers.append(NonBottleneck_1D(128, 0.3, 8, training))
			self.layers.append(NonBottleneck_1D(128, 0.3, 16, training))


	def inference(self, input, predict = False):

		output = self.initial_block.inference(input)
		for layer in self.layers:
			output = layer.inference(output)

		if predict:
			output = conv2d(output, 128, kernel_size = 1, strides = 1, padding = "SAME")

		return output

class UpsamplerBlock():

	def __init__(self, noutput, training):

		self.noutput = noutput
		self.training = training

	def inference(self, input):

		conv = conv2d_transpose(input, self.noutput, kernel_size = (3, 3), strides = (2, 2), padding = "SAME")
		bn = batch_normalization(conv, self.training)
		return tf.nn.relu(bn)

class Decoder():

	def __init__(self, num_classes = 1, training = True):

		self.num_classes = num_classes
		#self.training = training

		self.layers = []

		self.layers.append(UpsamplerBlock(64, training))
		self.layers.append(NonBottleneck_1D(64, 0, 1, training))
		self.layers.append(NonBottleneck_1D(64, 0, 1, training))

		self.layers.append(UpsamplerBlock(16, training))
		self.layers.append(NonBottleneck_1D(16, 0, 1, training))
		self.layers.append(NonBottleneck_1D(16, 0, 1, training))

	def inference(self, input):

		output = input
		for layer in self.layers:
			output = layer.inference(output)

		output = conv2d_transpose(output, self.num_classes, kernel_size = (3, 3), strides = 2, padding = "SAME", activation = None)

		return output
		
class Net():

	def __init__(self, num_classes = 1, training = True):

		#self.num_classes = 1
		self.training = training
		self.encoder = Encoder(num_classes)
		self.decoder = Decoder(num_classes)

	def __call__(self, input):

		return self.inference(input)

	def inference(self, input):

		output = self.encoder.inference(input)
		output = self.decoder.inference(output)

		return output

if __name__ == "__main__":

	#db = DownsamplerBlock(13, True)
	#a = tf.placeholder(tf.float32, [1, 1024, 512, 3])
	#b = db.inference(a)
	#print(b)
	#non = NonBottleneck_1D(3, 0.1, 1, True)
	#c = non.inference(a)
	#en = Encoder()
	#factor = en.inference(a)
	#print(factor)
	#UB = UpsamplerBlock(64, True)
	#print(UB.inference(factor))
	a = tf.placeholder(tf.float32, [1, 1024, 512, 3])
	net = Net(16)
	print(net(a))