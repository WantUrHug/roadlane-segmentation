import tensorflow as tf
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type = int, default = 300, help = "Number of epochs to train for")
parser.add_argument("--epoch_start_i", type = int, default = 0, help = "Start counting epochs from this number")
parser.add_argument("--validation_step", type = int, default = 5, help = "How often to save checkpoint(epochs)")
parser.add_argument("--checkpoint_step", type = int, default = 1, help = "How often to perform validation(epochs)")
parser.add_argument("--image", type = str, default = None, help = "The image you want to predict on.Only in 'predict' mode")
parser.add_argument("--datasets", type = str, default = "CamVid", help = "Dataset you are using")
parser.add_argument("--crop_height", type = int, default = 512, default = 512, help = "Height of cropped image into network")
parser.add_argument("--crop_width", type = int, default = 512, default = 512, help = "Width of cropped image into network")
parser.add_argument("--batch_size", type = int, default = 1, help = "Number of image in each epochs")
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')

args = parser.parse_args()
#print(args.num_epochs)

def data_augmentation(input_image, output_image):
	'''
	为什么是两张照片呢？因为输入和输出都是图片，尺寸是一样的！
	'''


	#对图片进行随机裁剪
	input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

	#如果设置了随机翻转，有50%的几率会旋转图片
	if args.h_flip and random.randint(0, 1):
		input_image = cv2.flip(input_image, 1)
		output_image = cv2.filp(output_image, 1)
	if args.v_flip and random.randint(0, 1):
		input_image = cv2.flip(input_image, 1)
		output_image = cv2.flip(output_image, 1)

	#灰度/明暗的变换
	if args.brightness:
		factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
		table = np.array([((i/255.0) * factor) * 255 for i in np.range(0, 256)]).astype(np.uint8)
		input_image = cv2.LUT(input_image, table)

	#图像增强中的旋转变换
	if args.rotation:
		angle = random.uniform(-1 * args.rotation, args.rotation)
	if args.rotation:
		M = cv2.GetRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
		input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flag = cv2.INTER_NEAREST)
		output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flag = cv2.INTER_NEAREST)

	return input_image, output_image


#从下面执行的内容来猜测，这个class_name_string是为了展示一共有多少个类，应对不同的数据集
class_name_list, label_values = helpers.get_label_info(args.datasets, "class_dict.csv")
class_name_string = ""
for class_name in class_name_list:
	if not class_name == class_name_list[-1]:
		class_name_string = class_name_string + class_name + ","
	else:
		class_name_string = class_name_string + class_name
#计算有多少个类
num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config = config)

#计算交叉熵
net_input = tf.placeholder(tf.float32, shape = [None, None, None, 3])
net_output = tf.placeholder(tf.float32, shape = [None, None, None, num_classes])

#自定义的模块，model_builder，内置了很多不同的解码和编码单元，利用这种外部传入参数的做法
#这个工程的架构更加规范
network, init_fn = model_builder.build_model(
	model_name = args.model,
	frontend = args.frontend,
	net_input = net_input,
	num_classes = num_classes,
	crop_width = args.crop_width,
	crop_height = args.crop_height,
	is_training = True)

#显然，network是模型计算的预测值，然后net_output是正确的数值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = network, labels = net_output))

opt = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 0.995).minimize(loss, var_list = [var for var in tf.trainable_variables()])

saver = tf.train.Saver(max_to_keep = 1000)
sess.run(tf.global_variables_initialzer())

#..............................?????????????????还没定义
utils.count_params()

if init_fn is nont None:
	init_fn(sess)

#加载一个之前的检查点，来继续训练
model_checkpoint_name = "checkpoint/latest_model_" + args.model + "_" + args.datasets + ".ckpt"
if args.continue_training:
	print("Loaded latest model checkpoint")
	saver.restore(sess, model_checkpoint_name)

#显然，利用自定义的函数加载数据
print("Loading the data......")
train_input_names, train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

val_indices = []
#前者是我们外部传入的参数，表示用多少张照片来做验证集；后者是从数据集中分割出来的，一共有多少张验证集的照片
#二者取小的，就是取有效的数量的意思
num_vals = min(args.num_val_images, len(val_input_names))

random.seed(16)
#val_input_names应该是一个字符串的数组，就是实际有的输入图片，len()计算有多少张，然后num_vals这个数肯定不大于
#len()的结果。例如 len() = 100, num_vals = 80, 所以这个random.sample就是不重复的从0-100(实际上最大就是取到99)
#中取出80个来组成我们最终验证集的内容
val_indices = random.sample(range(0, len(val_input_names)), num_vals)

for epoch in range(args.epochs_start_i, args.num_epochs):

	current_losses = []

	cnt = 0

	#相当于shuffle，把0，1，2....这样子序列给打乱
	id_list = np.random.permutation(len(train_input_names))

	#floor应该是向上取整，每个 epoch 都要历遍所有的数据，如果不整肯定就要额外多一次
	num_iters = int(np.floor(len(id_list)/args.batch_size))
	st = time.time()
	epoch_st = time.time() 

	for i in range(num_iters):

		input_image_batch = []
		output_image_batch = []

		for j in range(args.batch_size):

			index = i * args.batch_size + j
			id = id_list[index]
			input_image = utils.load_image(train_input_names[id])
			output_image = utils.load_image(train_output_names[id])

			with tf.device("/cpu:0"):
				input_image, output_image = data_augmentation(input_image, output_image)

				#输入图像需要归一化，然后输出的图像需要转化成one-hot向量
				#label_value 是我们从 csv 文件中提取出来的，我估计应该是对应了每种标签的数值
				#也就是名字所体现的 label-->value 的对应关系
				input_image = np.float32(input_image)/255.0
				output_image = np.float32(helpers.one_hot_it(label = output_image, label_values = label_values))

				#input_image 都是 512*512*3，然后为了形成一个batch，多一维出来，就需要把(512,512,3)给变成
				#(1,512,512,3),然后才可以加在一起.但需要注意，这一步还没成功，现在得到的是一个数组，还不是严格意义上的
				#np.ndarray，所以才需要有下一步的 stack .
				input_image_batch.append(np.expand_dims(input_image, axis = 0))
				output_image_batch.append(np.expand_dims(output_image, aixs = 0))

		#如果batch_size是1，那么上一步就是多余的，形成的(1,512,512,3)的张量没必要，直接拿出来作为(512,512,3)用就可以了
		if args.batch_size == 1:
			input_image_batch = input_image_batch[0]
			output_image_batch = output_image_batch[0]
		#否则，就要多一些处理?
		else:
			#np.squeeze函数，清除长度为1的维度(10,1,3)-->(10,3)
			#np.stack函数，将相同尺寸的张量连接起来，制造一个新的维度，axis就表示这个新维度是在第几个
			#例如我们有 10 个 (512,512,2),axis = 1,那么结果就是(512,10,512,2).但看起来有点不太符合情理，先这样
			input_image_batch = np.squeeze(np.stack(input_image_batch, axis = 1))
			output_image_batch = np.squeeze(np.stack(output_image_batch, axis = 1))
			#

		#开始训练
		_, current = sess.run([opt, loss], feed_dict = {net_input: input_image_batch, net_output: output_image_batch})
		current_losses.append(current)

		cnt = cnt + args.batch_size

		if cnt % 20 == 0:
			string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epochs, cnt, current, time.time() - st)
			utils.LOG(string_print)
			st = time.time()

	mean_loss = np.mean(current_losses)
	avg_loss_per_epoch.append(mean_loss)

	#创建一个目录
	if not os.path.isdir("%s/%04d"%("checkpoint", epoch)):
		os.mkdir("%s/%04d"%("checkpoint", epoch))
	#用相同的名字来保存，也就是覆盖了?每个epoch都要保存一下
	print("Saving latest checkpoint")
	saver.save(sess, model_checkpoint_name)

	if val_indices != 0 and epoch % args.checkpoint_step == 0:
		#为什么上面刚保存了一次 ckpt 又要再保存呢？没有重复的风险吗？？
		#但是是不重名的，不管了先理解这一步，就是要每多少个 epoch 保存一次模型
		print("Saving checkpoint for this epoch.")
		saver.save(sess, "%s/%04d/model.ckpt"%("checkpoints", epoch))

	if epoch % args。validation_step == 0:
		print("Performing validation")
		target = open("%s/%04d/val_scores.csv"%("checkpoints", epoch), "w")
		target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou, %s\n"%(class_name_string))

		scores_list = []
		class_scores_list = []
		precision_list = []
		recall_list = []
		f1_list = []
		iou_list = []

		#在一个小的数据样本群体上做验证，这里是逐张进行，后续可以改成按batch进行
		for ind in val_indices:

			input_image = np.expand_dims(np.float32(utils.load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]), axis = 0)/255.0
			gt = utils.load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
			gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

			#计算出输出
			output_image = sess.run(network, feed_dict = {net_input: input_image})

			#维度问题，四维张量其中第0维是1，本来也可以试试用np.squeeze
			output_image = np.array(output_image[0,:, :, :])
			output_image = helpers.reverse_one_hot(output_image)
			out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

			#这个函数后续再来定义。如果有多个类别，那么准确率accuracy，应该就是是否正确的一个指标，class_accuracies是不同类别各自的正确率吧
			#然后prec和其他的暂时还不是很理解
			accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred = output_image, label = gt, num_classes = num_classes)

			#应该是去掉绝对路径中的内容，只保留下名字
			file_name = utils.filepath_to_name(val_input_names[ind])
			target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))

			#item就是不同类别的准确率
			for item in class_accuracies:
				target.write(", %f"(item))
			target.write("\n")	

			scores_list.append(accuracy)
			class_scores_list.append(class_accuracies)
			precision_list.append(prec)
			recall_list.append(rec)
			f1_list.append(f1)
			iou_list.append(iou)

			#之前为了计算转化成one-hot张量，现在转化回去，为了可以保存成图片
			gt = helpers.colour_code_segmentation(gt, label_values)

			file_name = os.path.basename(val_input_names[ind])
			file_name = os.path.splitext(file_name)[0]
			cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints", epoch, file_name), cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_GRAY2BGR))
			cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints", epoch, file_name), cv2.cvtColor(np.uint8(gt), cv2.COLOR_GRAY2BGR))

		target.close()

		#提醒一下，现在还是在 validation 判定成功的条件下做的事情，现在统计的这些，是
		#在 validation 的数据上计算的分数
		avg_score = np.mean(scores_list)
		#这里要强调 axis = 0，因为不是一维的，而是二维的，其中的每个元素都代表了每张
		# validation 中的图片，在每个类别上的准确率。例如有5各类别，20张图片，那么
		#class_scores_list 可以看成是 20*5 的二维数组，对第一个维度求均值，也就是会得到 1*5
		#代表每个类别平均的准确率
		class_avg_scores = np.mean(class_scores_list, axis = 0)

		avg_scores_per_epoch.append(avg_score)
		avg_precision = np.mean(precisio_list)
		avg_recall = np.mean(recall_list)
		avg_f1 = np.mean(f1_list)
		avg_iou = np.mean(iou_list)
		avg_iou_per_epoch.append(avg_iou)

		print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
		print("Average per class validation accuracies for each epoch # %04d: "% (epoch))

		for index, item in enumerate(class_avg_scores):
			print("%s = %f"% (class_name_list[index], item))
		print("Validation precision = ", avg_precision)
		print("Validation recall = ", avg_recall)
		print("Validation F1 score = ", avg_f1)
		print("Validation IoU score = ", avg_iou)

	epoch_time = time.time() - epoch_st
	remain_time = epoch_time*(args.num_epochs - 1 - epoch)
	m, s = divmod(remain_time, 60)
	h, m = divmod(m, 60)
	if s != 0:
		train_time = "Remaining training time = %d hours %d minutes %d seconds\n"% (h, m, s)
	else:
		train_time = "Remaining training time = Training completed.\n"
	utils.LOG(train_time)
	scores_list = []
	fig1, ax1 = plt.subplots(figsize = (11, 8))
	ax1.plot(range(epoch + 1), avg_scores_per_epoch)
	ax1.set_title("Average validation accuracy vs epochs")
	ax1.set_xlabel("Epoch")
	ax1.set_ylabel("Current loss")
	plt.savefig("loss_vs_epochs.png")

	plt.clf()

	fig3, ax3 = plt.subplots(figsize=(11, 8))

    ax3.plot(range(epoch+1), avg_iou_per_epoch)
    ax3.set_title("Average IoU vs epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Current IoU")

    plt.savefig('iou_vs_epochs.png')