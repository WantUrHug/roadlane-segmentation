import os, shutil

#用 bat 处理完每张图片和标签都是各自一个文件夹
#先要把他们命名好然后存放到 data 和 label 对应
#的文件夹中
sparse_dir = "D:\\GitFile\\FuHua\\新建文件夹"
dst_dir = "D:\\GitFile\\roadlane-segmentation\\imgs"

#原本是先创建 data 和 label 文件夹，然后在他们之下再分别都创建
#train 和 test，后来发现这样子做不方便，因为在输入时需要成对给入网络
#如果 train_data 和 train_label 是在两个不同的路径下，不方便

#data_dir = os.path.join(dst_dir, "data")
#if not os.path.exists(data_dir):
#	os.mkdir(data_dir)
#data_train_dir = os.path.join(data_dir, "train")
#if not os.path.exists(data_train_dir):
#	os.mkdir(data_train_dir)
#data_test_dir = os.path.join(data_dir, "test")
#if not os.path.exists(data_test_dir):
#	os.mkdir(data_test_dir)
#
#label_dir = os.path.join(dst_dir, "label")
#if not os.path.exists(label_dir):
#	os.mkdir(label_dir)
#label_train_dir = os.path.join(label_dir, "train")
#if not os.path.exists(label_train_dir):
#	os.mkdir(label_train_dir)
#label_test_dir = os.path.join(label_dir, "test")
#if not os.path.exists(label_test_dir):
#	os.mkdir(label_test_dir)

train_dir = os.path.join(dst_dir, "train")
if not os.path.exists(train_dir):
	os.mkdir(train_dir)
train_data_dir = os.path.join(train_dir, "data")
train_label_dir = os.path.join(train_dir, "label")
if not os.path.exists(train_data_dir):
	os.mkdir(train_data_dir)
if not os.path.exists(train_label_dir):
	os.mkdir(train_label_dir)

test_dir = os.path.join(dst_dir, "test")
if not os.path.exists(test_dir):
	os.mkdir(test_dir)
test_data_dir = os.path.join(test_dir, "data")
test_label_dir = os.path.join(test_dir, "label")
if not os.path.exists(test_data_dir):
	os.mkdir(test_data_dir)
if not os.path.exists(test_label_dir):
	os.mkdir(test_label_dir)


train_size = 0.8
total = 0
item_namels = []
for item in os.listdir(sparse_dir):
	if item.endswith("_json"):
		total += 1
		item_name = item[:-5]
		item_namels.append(item_name)
print("You have prepare %d pictures and their labels.")

train_num = int(total * train_size)
test_num = total - train_num
print("Train number: %d, test number: %d."%(train_num, test_num))

cnt = 1
for item_name in item_namels:
	if cnt <= train_num:
		json_path = os.path.join(sparse_dir, item_name + "_json")
		img_path = os.path.join(json_path, "img.png")
		img_dst = os.path.join(train_data_dir, item_name + ".png")
		label_path = os.path.join(json_path, "label.png")
		label_dst = os.path.join(train_label_dir, item_name + ".png")
		shutil.copyfile(img_path, img_dst)
		shutil.copyfile(label_path, label_dst)
		cnt += 1
	else:
		json_path = os.path.join(sparse_dir, item_name + "_json")
		img_path = os.path.join(json_path, "img.png")
		img_dst = os.path.join(test_data_dir, item_name + ".png")
		label_path = os.path.join(json_path, "label.png")
		label_dst = os.path.join(test_label_dir, item_name + ".png")
		shutil.copyfile(img_path, img_dst)
		shutil.copyfile(label_path, label_dst)