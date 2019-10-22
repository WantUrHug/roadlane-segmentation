import numpy as np
import tensorflow as tf
import os

#导入自己定义的文件
import GCN
import utils

BATCH_SIZE = 10
IMG_W = 1920
IMG_H = 1056
IMG_C = 3

X = tf.placeholder(tf.float16, [None, IMG_W, IMG_H, IMG_C])
Y = tf.placeholder(tf.float16, [None, IMG_W, IMG_H, IMG_C])
_pred = GCN.build_gcn(X, 2)

print(_pred)