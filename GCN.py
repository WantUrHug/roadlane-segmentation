import tensorflow as tf
from tensorflow.layers import conv2d, conv2d_transpose, max_pooling2d
#tensorflow.nn.conv2d是没有附带 relu的,而 tensorflow.layers.conv2d是默认使用 relu 作为激活函数的
#conv2d(input, fitler, strides, padding, dilation)
import numpy as np
import os

def DownSampler(input, num_output):
	'''
	下采样单元，用卷积层和池化层拼接即可
	'''

	net = conv2d(input, num_output, kernel_size = (3, 3), strides = (1, 1), padding = "SAME")
	net = max_pooling2d(net, pool_size = (2,2), strides = 2)

	return net

def GCNblock(input, num_classes, size = 3):
	'''
	使用两组空间可分离卷积来减少参数量
	'''

	net1 = conv2d(input, num_classes, kernel_size = (1, size), strides = (1, 1), padding = "SAME")
	net1 = conv2d(net1, num_classes, kernel_size = (size, 1), strides = (1, 1), padding = "SAME")

	net2 = conv2d(input, num_classes, kernel_size = (size, 1), strides = (1, 1), padding = "SAME")
	net2 = conv2d(net2, num_classes, kernel_size = (1, size), strides = (1, 1), padding = "SAME")
	return tf.add(net1, net2)

def BRblock(input):
	'''
	Boundary refinement, 简单的残差链接
	'''
	num_classes = input.shape.as_list()[-1]
	net = conv2d(input, num_classes, kernel_size = (3, 3), strides = (1 ,1), activation = "relu", padding = "SAME")
	net = conv2d(net, num_classes, kernel_size = (3, 3), strides = (1, 1), padding = "SAME")

	return tf.add(input, net)

def UpSampler(input, num_outputs = None):
	'''
	上采样单元，使用转置卷积来使尺寸扩大两倍
	'''
	if not num_outputs:
		num_outputs = input.shape.as_list()[-1]
	net = conv2d_transpose(input, num_outputs, kernel_size = (3, 3), strides = (2, 2), padding = "SAME")

	return net

def built_gcn(input, num_classes = 2):
	'''
	构建整个网络结构.
	'''
	#DS = Down Samplper，先逐层降采样
	DS1 = DownSampler(input, 10)
	DS2 = DownSampler(DS1, 15)
	DS3 = DownSampler(DS2, 20)
	DS4 = DownSampler(DS3, 30)

	#根据逻辑先从深的层开始处理
	GCN4 = GCNblock(DS4, num_classes)
	BR4 = BRblock(GCN4)
	#US = Up Sampler，上采样
	US4 = UpSampler(BR4)

	GCN3 = GCNblock(DS3, num_classes)
	BR3 = BRblock(GCN3)
	#CON3 = tf.concat([BR3, US4], 0)
	ADD3 = BR3 + US4
	BR3 = BRblock(ADD3)
	US3 = UpSampler(BR3)

	GCN2 = GCNblock(DS2, num_classes)
	BR2 = BRblock(GCN2)
	#CON2 = tf.concat([BR2, US3], 0)
	ADD2 = BR2 + US3
	BR2 = BRblock(ADD2)
	US2 = UpSampler(BR2)

	GCN1 = GCNblock(DS1, num_classes)
	BR1 = BRblock(GCN1)
	#CON1 = tf.concat([BR1, US2], 0)
	ADD1 = BR1 + US2
	BR1 = BRblock(ADD1)
	US1 = UpSampler(BR1)

	return BRblock(US1)

if __name__ == "__main__":

	a = tf.placeholder(tf.float32, [10, 1920, 1056, 3])
	b = built_gcn(a)
	print(b)
