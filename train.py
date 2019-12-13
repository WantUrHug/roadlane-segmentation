import numpy as np
import tensorflow as tf
import os, time

#导入自己定义的文件
import GCN
import utils

os.environ['CUDA_VISIBLE_DEVICES']='0'

BATCH_SIZE = 8
#不使用原尺寸，选择 RANDOM_CROP
IMG_W = 640
IMG_H = 1280
IMG_C = 3
NUM_CLASSES = 1

MAXSTEP = 2000
CHECK_STEP = 10
SAVE_STEP = 400

train_dir = "D:\\GitFile\\roadlane-segmentation-imgs\\train"
test_dir = "D:\\GitFile\\roadlane-segmentation-imgs\\test"
model_dir = "D:\\GitFile\\roadlane-segmentation\\model"
train_data_handle = utils.handle_getter(train_dir, IMG_H, IMG_W)(utils.get_data)
test_data_handle = utils.handle_getter(test_dir, IMG_H, IMG_W)(utils.get_data)

train_dataset = tf.data.Dataset.from_generator(train_data_handle, output_types = (tf.float32, tf.float32))
train_dataset = train_dataset.shuffle(20).batch(BATCH_SIZE, drop_remainder = True).repeat()
train_iterator = train_dataset.make_one_shot_iterator()
next_train_data, next_train_label = train_iterator.get_next()

test_dataset = tf.data.Dataset.from_generator(test_data_handle, output_types = (tf.float32, tf.float32))
test_dataset = test_dataset.shuffle(20).batch(BATCH_SIZE, drop_remainder = True).repeat()
test_iterator = test_dataset.make_one_shot_iterator()
next_test_data, next_test_label = test_iterator.get_next()

print("*******************************************************")
print("* Finish step 1: preparing data. *")


X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C], name = "X")
Y = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, NUM_CLASSES])
_pred = GCN.build_gcn(X, NUM_CLASSES)
tf.add_to_collection("infrence", _pred)

#logits 是前向传播的结果，labels 是正确的标签
#print(_pred.shape, Y.shape)
_pred_flatten = _pred#utils.label_before_cal_loss(_pred)
Y_flatten = Y#utils.label_before_cal_loss(Y)
#_pred_flatten = tf.concat()
#print(_pred_flatten[0].shape)
#print("len of prepare: ", utils.label_before_cal_loss(_pred).shape) #tf.layers.Flatten()(_pred)
#Y_flatten = tf.layers.Flatten()(Y)
#print(_pred_flatten.shape)
#print(Y_flatten.shape)
#softmax = tf.nn.softmax_cross_entropy_with_logits(logits = _pred_flatten, labels = Y_flatten)
loss = utils.weighted_loss_v1(logits = _pred_flatten, labels = Y_flatten)#tf.reduce_mean(softmax)
acc = utils.total_accuracy(logits = _pred_flatten, labels = Y_flatten)
#print(loss.shape)

#print("1")
global_step = tf.Variable(0, trainable = False)
#print("2")
opt = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss, global_step = global_step)
#print("3")
#global_acc = utils.cal_global_accuracy(_pred, Y)
#print("4")
#history = {}
#history["train_loss"] = []
#history["test_loss"] = []
#history["train_acc"] = []
#history["test_acc"] = []

config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
config.gpu_options.allow_growth = True
#print("aaaaaaaaaaa")
#assert 1==0
with tf.Session(config = config) as sess:
	print("*******************************************************")
	print("* Start session.*")
	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver(max_to_keep = 10)

	time_cost = 0
	start_time = time.time()

	for step in range(1, MAXSTEP + 1):

		train_data, train_label = sess.run((next_train_data, next_train_label))
		with tf.device("/cpu:0"):
			train_data = (np.float32(train_data)/255.0-0.5)*2
		
		if step % CHECK_STEP == 0:
			test_data, test_label = sess.run((next_test_data, next_test_label))
			with tf.device("/cpu:0"):
				test_data = (np.float32(test_data)/255.0-0.5)*2
			_, train_loss_value, train_acc_value = sess.run((opt, loss, acc), feed_dict = {X: train_data, Y: train_label})
			_, test_loss_value, test_acc_value = sess.run((opt, loss, acc), feed_dict = {X: test_data, Y: test_label})
			#print(train_acc_value, train_loss_value)
			#history["train_loss"].append(train_loss_value)
			#history["train_acc"].append(train_acc_value)
			#history["test_loss"].append(test_loss_value)
			#history["test_acc"].append(test_acc_value)

			time_cost = time.time() - start_time
			start_time = time.time()

			print("Step %d, time cost: %.1fs, train loss: %.2f, train acc: %.2f%%, test loss: %.2f, test_acc: %.2f%%." % (step, time_cost, train_loss_value, train_acc_value*100, test_loss_value, test_acc_value*100))
			#print("Step %d, time cost: %.1fs, train loss: %.2f, test loss: %.2f." % (step, time_cost, train_loss_value, test_loss_value))

		else:
			sess.run(opt, feed_dict = {X: train_data, Y: train_label})
		if step%SAVE_STEP == 0:
			saver.save(sess, os.path.join(model_dir, "gcn"), global_step = step)