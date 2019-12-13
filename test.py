import tensorflow as tf
import numpy as np
import os
from PIL import Image

def test_pic(img_path, model_dir):

	ckpt = tf.train.get_checkpoint_state(model_dir)
	lastest_model = ckpt.model_checkpoint_path

	img = Image.open(img_path)
	img = img.resize((640, 1280))
	#img.show()

	with tf.Session() as sess:
		print()
		saver = tf.train.import_meta_graph("%s.meta"%lastest_model.split('/')[-1])
		
		saver.restore(sess, lastest_model)

		graph = tf.get_default_graph()
		#tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
		#for tensor_name in tensor_name_list:
		#	print(tensor_name,'\n')


		X = graph.get_tensor_by_name("X:0")
		inference = tf.get_collection("infrence")[0]

		#img = np.array(img, dtype = np.float32)/255.0
		img = (np.array(img, dtype = np.float32)/255.0-0.5)*2
		output = sess.run(inference, feed_dict = {X: [img,img,img,img,img,img,img,img]})

		return tf.argmax(output[0], 2).eval(session = sess)

def label2img(label, label_value = [[0, 0, 0], [255, 255, 255]]):
	'''
	将多标签通道图给转化成二值化的图像来观察。选择 #FFFFFF 白色更直观。
	'''

	cnt = 0
	output = np.zeros([label.shape[0], label.shape[1], 3], dtype = np.uint8)
	for i in range(output.shape[0]):
		for j in range(output.shape[1]):
			if label[i][j] == 1:
				cnt += 1
			output[i, j] = label_value[label[i][j]]
	print("red cnt:", cnt)

	return output

if __name__ == "__main__":
	#test_dir = "D:\\GitFile\\roadlane-segmentation-imgs\\test\\data\\"
	#test_dir = "E:\\train"
	for filename in ["D:\\YZlogs\\20191212\\31.png"]:#os.listdir(test_dir):
		#im = os.path.join(test_dir, filename)
		output = test_pic(filename, "model\\v3")
	#print(output.shape)
		out= label2img(output)
	#print(out.shape)
	#cv2.imshow(out, "output")
	#cv2.waitKeys(0)
	#print(out)
		img = Image.fromarray(out)
		img.show()
		#img.save("output1\\output_%s.jpg"%filename)