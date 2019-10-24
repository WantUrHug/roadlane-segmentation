import os, shutil

#用 bat 处理完每张图片和标签都是各自一个文件夹
#先要把他们命名好然后存放到 data 和 label 对应
#的文件夹中
sparse_dir = "D:\\GitFile\\FuHua\\新建文件夹"
dst_dir = "D:\\GitFile\\roadlane-segmentation\\imgs"

data_dir = os.path.join(dst_dir, "data")
if not os.path.exists(data_dir):
	os.mkdir(data_dir)
data_train_dir = os.path.join(data_dir, "train")
if not os.path.exists(data_train_dir):
	os.mkdir(data_train_dir)
data_test_dir = os.path.join(data_dir, "test")
if not os.path.exists(data_test_dir):
	os.mkdir(data_test_dir)

label_dir = os.path.join(dst_dir, "label")
if not os.path.exists(label_dir):
	os.mkdir(label_dir)
label_train_dir = os.path.join(label_dir, "train")
if not os.path.exists(label_train_dir):
	os.mkdir(label_train_dir)
label_test_dir = os.path.join(label_dir, "test")
if not os.path.exists(label_test_dir):
	os.mkdir(label_test_dir)


train_size = 79
test_size = 20
i = 0
for item in os.listdir(sparse_dir):
	if item.endswith("_json"):
		#print(item)
		item_name = item[:-5]
		#print(item_name)
		if i <= 79:
			json_path = os.path.join(sparse_dir, item)
			img_path = os.path.join(json_path, "img.png")
			img_dst = os.path.join(data_train_dir, item_name + ".png")
			label_path = os.path.join(json_path, "label.png")
			label_dst = os.path.join(label_train_dir, item_name + ".png")
			shutil.copyfile(img_path, img_dst)
			shutil.copyfile(label_path, label_dst)
			i += 1
		else:
			json_path = os.path.join(sparse_dir, item)
			img_path = os.path.join(json_path, "img.png")
			img_dst = os.path.join(data_test_dir, item_name + ".png")
			label_path = os.path.join(json_path, "label.png")
			label_dst = os.path.join(label_test_dir, item_name + ".png")
			shutil.copyfile(img_path, img_dst)
			shutil.copyfile(label_path, label_dst)