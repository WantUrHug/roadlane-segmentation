import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os, subprocess, time

#tf.enable_eager_execution()

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
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (640, 1280))
		label = cv2.imread(os.path.join(label_dir, item), -1)
		label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
		label = cv2.resize(label, (640, 1280))
		#image = cv2.resize(image, (0.5, 0.5))
		#label = cv2.resize(image, (0.5, 0.5))
		image, label = data_augumentation((image, label), (image_h, image_w))
		#shape = image.shape
		#if not (shape[0] == image_h and shape[1] == image_w):
		#	image = cv2.reshape(image, (image_h, image_w))
		#	label = cv2.reshape(label, (image_h, image_h))
		yield image, to_one_hot(label, label_values, boolean = False)

def data_augumentation(input, output_size, method = "RANDOM_FLIP"):

	if method == "RANDOM_CROP":
		image, label = input
		input_shape = image.shape
		input_h, input_w = image.shape[0:2]
		output_h, output_w = output_size
		random_h = np.random.randint(0, input_h - output_size[0])
		random_w = np.random.randint(0, input_w - output_size[1])

		return image[random_h: random_h + output_h, random_w: random_w + output_w, :], label[random_h: random_h + output_h, random_w: random_w + output_w, :]
	elif method == "RANDOM_FLIP":
		image, label = input
		if np.random.randint(2):#0 or 1
			image = cv2.flip(image, 0)
			label = cv2.flip(label, 0)
		if np.random.randint(2):
			image = cv2.flip(image, 1)
			label = cv2.flip(label, 1)
		return image, label


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
	原本是 depth = 2，如今觉得既然是二分类那么只需要有一个标的物的概率就可以了，有时两个数值
	反而对于一些指标的计算更加困难。
	boolean：可以选择输出的结果是布尔值还是数值
	'''

	#semantic_map = []
	#if boolean:
	#	for colour in label_values:
	#		equality = np.equal(label, colour)
	#		class_map = np.all(equality, axis = -1)
	#		semantic_map.append(class_map)
	#	semantic_map = np.stack(semantic_map, axis = -1)
	#	return semantic_map
	#else:	
	#	for colour in label_values:
	#		equality = np.equal(label, colour)
	#		out = np.zeros(equality.shape[:2], dtype = np.float)
	#		class_map = np.all(equality, axis = -1, out = out)
	#		semantic_map.append(out)
	#	semantic_map = np.stack(semantic_map, axis = -1)
	#	return semantic_map	
	#print(label.shape)
	if boolean:
		equality = np.equal(label, label_values[1])
		class_map = np.all(equality, axis = -1)
		return np.expand_dims(class_map, axis = -1)
	else:
		equality = np.equal(label, label_values[1])
		out = np.zeros(equality.shape[:2], dtype = np.float)
		class_map = np.all(equality, axis = -1, out = out)
		return np.expand_dims(class_map, axis = -1)
	
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

def weighted_loss_v1(logits, labels, weight = [20.0, 1]):
	'''
	原先使用的损失函数，后续发现，语义分割模型不适用交叉熵这个适用于分类的损失函数，
	所以导致使用后的效果也挺差的。最好是用，每个像素点的预测类是否与真实类相同，相同为0
	不同为1，然后乘上对应的权重，作为一个最小化的目标。
	而且在使用的时候容易变成 nan，原因不详
	'''
	#weight = tf.constant(weight)#(2,)
	labels = tf.rint(labels)
	loss = -(tf.log(logits + 10E-6)*labels*weight[0] + tf.log(1 - logits + 10E-6)*(1 - labels)*weight[1]) 
	#loss = loss*weight
	loss = tf.reduce_mean(loss)
	#之前的版本，因为改了一下改成了

	return loss

def weighted_loss_v2(logits, labels, weight = [1.0, 30.0]):

	logits = tf.rint(logits)#把介于 0 和 1之间的概率值转化为更接近的整数，0 或者 1

	#如果 logits = 0, label = 1，那么说明是判断错了，原本是线但是没有找出来，loss 就是下面第一项
	#如果 logits = 1, label = 0，也是判断错了，原本是北京但是没找出来，loss 是对应着第二项
	#如果是 0、0 或者 1、1 的情况那么就这一个像素点所提供的损失是 0
	print(logits.shape, labels.shape)
	ones = tf.ones_like(logits)
	loss= (ones - logits)*labels*weight[0] + logits*(ones - labels)*weight[1]
	loss = tf.reduce_mean(loss)

	return loss

def total_accuracy(logits, labels):
	'''
	全局准确率.修改之后 depth 只有1，也就是标的物的概率.
	'''
	logits = tf.rint(logits)

	correct = tf.equal(logits, labels)
	acc = tf.reduce_mean(tf.cast(correct, tf.float32))

	return acc

def _count(label, label_value = [[0, 0, 0], [128, 0, 0]]):
	cnt = 0
	for i in range(label.shape[0]):
		for j in range(label.shape[1]):
			if label[i, j, 0] == 128:
				cnt += 1

	print(cnt)

if __name__ == "__main__":

	test_label = "D:\\GitFile\\roadlane-segmentation-imgs\\train\\label\\1.png"
	i = cv2.imread(test_label, -1)
	i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
	#print(i)
	r = to_one_hot(i, boolean = False)
	print(r.shape)
	#_count(i)

	