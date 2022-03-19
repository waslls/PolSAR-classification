import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import os
import cv2
import copy
from utils.data_procession import convert

def get_data(args):
    feature_names=next(os.walk(args.feature_folder))[2]
    # print(feature_names)
    feature = []
    for i in range(len(feature_names)):
        feature.append(cv2.cvtColor(mpimg.imread(args.feature_folder + "/%s" % feature_names[i]), cv2.COLOR_RGB2GRAY))
    data = np.array(feature).transpose((1, 2, 0))#(124, 750, 1024)->(750, 1024, 124)
    data = data/255 #data.dtype:dtype('float64')
   
    data_train = data#训练数据
    data_all = copy.deepcopy(data)#用于生成结果
    label = get_label(args)
    data_train[label==0]=0
    return data_train, data_all, label
    
def get_label(args):
    label = tf.io.read_file(args.label_folder)
    label = np.array(tf.image.decode_png(label))
    label = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
    label = convert(label, np.unique(label)[1:])#生成(0,15)的标签(750，1024)
    #print(np.unique(label))#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
    return label
    
    
