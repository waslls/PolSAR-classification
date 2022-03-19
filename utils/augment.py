import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def rotate(xb, yb, angle, img_w, img_h):
    # cv2.getRotationMatrix2D()，这个函数需要三个参数，旋转中心，旋转角度，旋转后图像的缩放比例
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
    # cv2.warpAffine()仿射变换，参数src - 输入图像  M - 变换矩阵(一般反映平移或旋转的关系)  dsize - 输出图像的大小  flags - 插值方法的组合 。。。
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb

def add_noise_255(img):
    img = np.array(img)
    img_h, img_w, _ = img.shape
    for i in range((img_h*img_w)//200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])  # h
        temp_y = np.random.randint(0, img.shape[1])  # w
        img[temp_x][temp_y] = 255
    return img

def add_noise_0(img):
    img = np.array(img)
    img_h, img_w, _ = img.shape
    for i in range((img_h*img_w)//200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])  # h
        temp_y = np.random.randint(0, img.shape[1])  # w
        img[temp_x][temp_y] = 0
    return img
# def flipLabelUpDown(label):
#     rows, cols = np.array(label).shape
#     for i in range(rows//2):
#         for j in range(cols):
#             temp = label[i][j]
#             label[i][j] = label[rows - 1 - i][j]
#             label[rows - 1 - i][j] = temp
#     return label
# def flipLabelLeftRight(label):
#     rows, cols = np.array(label).shape
#     for i in range(rows):
#         for j in range(cols//2):
#             temp = label[i][j]
#             label[i][j] = label[i][cols-1-j]
#             label[i][cols-1-j] = temp
#     return label

def data_augments_brightsness(data_train):
    data_train = np.array(data_train)
    number, _, _, _ = data_train.shape
    data_augment = []
    # print(data_augment.shape)#64 64 3
    for i in range(number):
        image = tf.image.random_brightness(data_train[i], max_delta=0.5)
        # print(image.shape)#64 64 3
        data_augment.append(image)
        # data_augment = np.array(data_augment)
        # print(data_augment.shape)
    return data_augment
# 随机改变对比度
def data_augments_contrast(data_train):
    data_train = np.array(data_train)
    number, _, _, _ = data_train.shape
    data_augment = []
    for i in range(number):
        image = tf.image.random_contrast(data_train[i], 0.2, 0.7)
        data_augment.append(image)
    return data_augment

def data_augments_saturation(data_train):
    data_train = np.array(data_train)
    number, _, _, _ = data_train.shape
    data_augment = []
    for i in range(number):
        image = tf.image.random_saturation(data_train[i], 0, 5)
        data_augment.append(image)
    return data_augment
# 随机上下翻转
def data_label_augments_updown_flip(data_train, label):
    data_train = np.array(data_train)
    label = np.array(label)
    number, _, _, _ = data_train.shape
    data_augment = []
    label_augment = []
    for i in range(number):
        image = cv2.flip(data_train[i], 0)
        data_augment.append(image)
        label_updown = cv2.cvtColor(label[i], cv2.COLOR_GRAY2BGR)
        label_updown = cv2.flip(label_updown, 0)
        label_updown = np.array(label_updown)
        label_updown = cv2.cvtColor(label_updown, cv2.COLOR_RGB2GRAY)
        label_augment.append(label_updown)
    return data_augment, label_augment
# 随机左右翻转
def data_label_augments_lr_flip(data_train, label):
    data_train = np.array(data_train)
    label = np.array(label)
    number, _, _, _ = data_train.shape
    data_augment = []
    label_augment = []
    for i in range(number):
        image = cv2.flip(data_train[i], 1)
        data_augment.append(image)
        label_lr = cv2.cvtColor(label[i], cv2.COLOR_GRAY2BGR)
        label_lr = cv2.flip(label_lr, 1)#tf.image.flip_left_right(label_lr)
        label_lr = np.array(label_lr)
        label_lr = cv2.cvtColor(label_lr, cv2.COLOR_RGB2GRAY)
        label_augment.append(label_lr)
    return data_augment, label_augment
# def label_augments_lr_flip(label):
#     label = np.array(label)
#     number, _, _ = label.shape
#     label_augment = []
#     for i in range(number):
#         image = flipLabelLeftRight(label[i])
#         label_augment.append(image)
#     return label_augment
#
# def label_augments_updown_flip(label):
#     label = np.array(label)
#     number, _, _ = label.shape
#     label_augment = []
#     for i in range(number):
#         image = flipLabelUpDown(label[i])
#         label_augment.append(image)
#     return label_augment

def data_augment_rotate(data_train, label):
    data_train = np.array(data_train)
    label = np.array(label)
    number, img_w, img_h, _ = data_train.shape
    data_augment = []
    label_augment = []
    for i in range(number):
        label_rgb = cv2.cvtColor(label[i], cv2.COLOR_GRAY2BGR)#rotate里的函数只能接受rgb三通道图像
        image_90, label_90 = rotate(data_train[i], label_rgb, 90, img_w, img_h)
        label_90 = cv2.cvtColor(label_90, cv2.COLOR_RGB2GRAY)#标签需要灰度图
        data_augment.append(image_90)
        label_augment.append(label_90)
        image_180, label_180 = rotate(data_train[i], label_rgb, 180, img_w, img_h)
        label_180 = cv2.cvtColor(label_180, cv2.COLOR_RGB2GRAY)
        data_augment.append(image_180)
        label_augment.append(label_180)
        image_270, label_270 = rotate(data_train[i], label_rgb, 270, img_w, img_h)
        label_270 = cv2.cvtColor(label_270, cv2.COLOR_RGB2GRAY)
        data_augment.append(image_270)
        label_augment.append(label_270)
    return data_augment, label_augment

def data_augment_add_noise(data_train, label):
    data_train = np.array(data_train)
    number, _, _, _ = data_train.shape
    data_augment = []
    label_augment = []
    for i in range(number):
        img_0 = add_noise_0(data_train[i])
        img_1 = add_noise_255(data_train[i])
        data_augment.append(img_0)
        label_augment.append(label[i])
        data_augment.append(img_1)
        label_augment.append(label[i])
    return data_augment, label_augment
