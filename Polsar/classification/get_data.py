import os
import cv2
import numpy as np
import matplotlib.image as mpimg

def convert(a, b):  # 以前为0的保持为0
    c = np.array(b)
    q = c.shape[0]
    total_num, a_height = a.shape  # 750 1024
    for i in range(total_num):
        for j in range(a_height):
            for z in range(q):
                if a[i][j] == b[z]:
                    a[i][j] = z + 1
    return a

def get_data(args):
    # 获取数据和标签
    feature_names = next(os.walk(args.feature_folder))[2]
    feature = []
    for i in range(len(feature_names)):
        feature.append(cv2.cvtColor(mpimg.imread(args.feature_folder + "/%s" % feature_names[i]), cv2.COLOR_RGB2GRAY))
    data = np.array(feature).transpose((1, 2, 0))
    label = cv2.imread(args.label_folder, 0)
    label = convert(label, np.unique(label)[1:])  # 生成（0，15）的标签 size=（750，1024）
    return data, label