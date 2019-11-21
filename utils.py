import tensorflow as tf
import numpy as np
import cv2
import os, subprocess, time

def get_data(train_dir, image_h, image_w):
	'''
	从文件夹中获取一张图片
	'''
	#cwd = os.getcwd()
	#data_dir = os.path.join(cwd, "imgs")

	data_dir = os.path.join(train_dir, "data")
	label_dir = os.path.join(train_dir, "label")

	#这里的顺序是 RGB .
	label_values = [[0, 0, 0], [128, 0, 0]]

	#两个文件夹中对应的data和label，名字都是一样的以 png 结尾
	for item in os.listdir(data_dir):
		#item_dir = os.path.join(train_data, i)
		image = cv2.imread(os.path.join(data_dir, item), -1)
		label = cv2.imread(os.path.join(label_dir, item), -1)
		#image = cv2.resize(image, (0.5, 0.5))
		#label = cv2.resize(image, (0.5, 0.5))
		image, label = data_augumentation((image, label), (image_h, image_w), ".")
		#shape = image.shape
		#if not (shape[0] == image_h and shape[1] == image_w):
		#	image = cv2.reshape(image, (image_h, image_w))
		#	label = cv2.reshape(label, (image_h, image_h))
		yield image, to_one_hot(label, label_values, boolean = False)

def data_augumentation(input, output_size, method = "RANDOM_CROP"):

	if method == "RANDOM_CROP":
		image, label = input
		input_shape = image.shape
		input_h, input_w = image.shape[0:2]
		output_h, output_w = output_size
		random_h = np.random.randint(0, input_h - output_size[0])
		random_w = np.random.randint(0, input_w - output_size[1])

		return image[random_h: random_h + output_h, random_w: random_w + output_w, :], label[random_h: random_h + output_h, random_w: random_w + output_w, :]
	else:
		return input


def handle_getter(*args, **kwargs):
	'''
	用另一种办法来为 get_train_data 函数提供参数.
	'''
	def outer_wrapper(fun):
		def inner_wrapper():
			return fun(*args, **kwargs)
		return inner_wrapper
	return outer_wrapper

def to_one_hot(label, label_values = [[0, 0, 0], [128, 0, 0]], boolean = True):
	'''
	一种方法是得到一个由 True False 组成的矩阵，但我不确定这样子可以与其他
	以数值为元素的矩阵进行计算吗？
	'''
	semantic_map = []
	if boolean:
		for colour in label_values:
			equality = np.equal(label, colour)
			class_map = np.all(equality, axis = -1)
			semantic_map.append(class_map)
		semantic_map = np.stack(semantic_map, axis = -1)
		return semantic_map
	else:	
		for colour in label_values:
			equality = np.equal(label, colour)

			out = np.zeros(equality.shape[:2], dtype = np.float)
			class_map = np.all(equality, axis = -1, out = out)
			semantic_map.append(out)
		semantic_map = np.stack(semantic_map, axis = -1)
		return semantic_map
	
	#print(label.shape)
	'''
	shape1, shape2 = label.shape[:2] 
	result = np.zeros([shape1, shape2, len(label_values)])
	for i in range(shape1):
		for j in range(shape2):
			for k in range(len(label_values)):
				if (label[i][j] == label_values[k]).all():	
					result[i, j][k] = 1
					break
	return result
	'''
def label_before_cal_loss(input):
	'''
	将图片的 label 以及网络计算输出的结果，给到计算交叉熵之前，需要把原本四维的
	张量转化成二维的张量》
	'''
	output = input
	while len(output.shape) > 2:
		output = tf.unstack(output, axis = 0)
		output = tf.concat(output, axis = 0)
	return output


def cal_global_accuracy(logits, labels):
	'''
	对给入的 batch_size 数据进行计算，计算它们的 global_accuracy.
	'''
	batch_size, h, w, _ = labels.shape.as_list()
	#print(batch_size)
	total = batch_size*h*w
	cnt = 0
	for i in range(batch_size):
		for j in range(h):
			for k in range(w):
				if logits[i][j][k] == labels[i][j][k]:
					cnt += 1
	return float(cnt)/float(total)

def weighted_loss(logits, labels, weight = [1.0, 10.0]):

	weight = tf.constant(weight)/sum(weight)
	loss = -tf.log(tf.nn.softmax(logits))*labels
	loss = loss*weight
	loss = tf.reduce_mean(loss)

	return loss

def accuracy(logits, labels):

	correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	acc = tf.reduce_mean(tf.cast(correct, tf.float32))

	return acc

if __name__ == "__main__":
	test_label = "D:\\GitFile\\roadlane-segmentation\\imgs\\train\\label\\00001.png"
	i = cv2.imread(test_label, -1)
	i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
	r = to_one_hot(i)
	print(r)

	