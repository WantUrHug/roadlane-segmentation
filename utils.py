import tensorflow as tf
import numpy as np
import cv2
import os, subprocess, time

def get_train_data(batch_size, image_h, image_w):
	cwd = os.getcwd()
	data_dir = os.path.join(cwd, "imgs")

	train_data = os.path.join(data_dir, "train_data")
	#validation_dir = os.path.join(data_dir, "validation")

	#这里的顺序是 RGB .
	label_values = [[0, 0, 0], [128, 0, 0]]

	for i in os.listdir(train_dir):
		item_dir = os.path.join(train_data, i)
		image = cv2.imread(os.path.join(item_dir, "img.png"), -1)
		label = cv2.imread(os.path.join(item_dir, "label.png"), -1)
		shape = image.shape
		if shape[0] == image_h and shape[1] == image_w:
			image = cv2.reshape(image, (image_h, image_w))
			label = cv2.reshape(label, (image_h, image_h))
		return image, to_one_hot(label, label_values)

def to_one_hot(label, label_values = [[0, 0, 0], [128, 0, 0]]):
	semantic_map = []
	for colour in label_values:
		equality = np.equal(label, colour)
		class_map = np.all(equality, axis = -1)
		semantic_map.append(class_map)
	semantic_map = np.stack(semantic_map, axis = -1)
	return semantic_map

if __name__ == "__main__":
	test_label = "D:\\YZlogs\\20190923\\label\\5_json\\label.png"
	i = cv2.imread(test_label, -1)
	i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
	r = to_one_hot(i)
	print(r.shape)

	