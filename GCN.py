import tensorflow as tf
from tensorflow.layers import conv2d, maxpool2d
import numpy as np
import os

def DownSampler(input, num_output):
	'''
	下采样单元，用卷积层和池化层拼接即可
	'''

	net = conv2d(input, num_output, kernel_size = (3,3), strides = 1, padding = "SAME")
	net = maxpool2d(net, kernel_size = (2,2), strides = 2)

	return net

def GCNblock(input, num_classes, size):
 	
 	net1 = conv2d(input, num_classes, kernel_size = (1, 3), strides = 1, padding = "SAME")
 	net1 = conv2d(net1, num_classes, kernel_size = (3, 1), strides = 1, padding = "SAME")

 	net2 = conv2d(input, num_classes, kernel_size = (3, 1), strides = 1, padding = "SAME")
	net2 = conv2d(net2, num_classes, kernel_size = (1, 3), strides = 1, padding = "SAME")

	return tf.add(net1, net2)

def BRblock(input, )