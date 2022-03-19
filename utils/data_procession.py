import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#https://blog.csdn.net/qq_40549291/article/details/85274581
import numpy as np
import tensorflow as tf

#将图片补零以让其为正方形  原始[750/900,1024,3]  target_length必须等于图片的长或宽
def padding(a, target_length):#
    a_height, a_width, a_channel = a.shape
    if a_height < target_length:
        b = [[[0 for channel in range(a_channel)] for col in range(a_width)]for row in range(target_length-a_height)]
        a = np.concatenate((a, b), axis=0)
    elif a_width < target_length:
        b = [[[0 for channel in range(a_channel)] for col in range(target_length-a_width)]for row in range(a_height)]
        a = np.concatenate((a, b), axis=1)
    return a

#原图切为大小为训练时的patch来做预测  a需为方形且a的长宽为child_length的倍数
def cut_image(a,child_length):
    a_height, a_width, _ = a.shape
    data = []
    num = a_height//child_length
    for i in range(num):
        for j in range(num):
            b = a[i*child_length:(i+1)*child_length, j*child_length:(j+1)*child_length]
            data.append(b)
    return data

#每个patch的预测结果拼为大图
#将数据的第一维视为样本个数，将样本以行优先拼成一张大图 输入数据维度（1024.32.32）如果第四维是1 将其squeeze掉
def combination(a):
    total_num,a_height,a_width = a.shape #64 128 128
    num = np.sqrt(total_num)#8
    num = int(num)
    row_num = num * a_height#8*128=1024
    col_num = num * a_width
    data = [[0 for col in range(col_num)] for row in range(row_num)]#[1024,1024]
    data = np.array(data)
    for i in range(num):
        for j in range(num):
            data[i*a_height:(i+1)*a_height, j*a_width:(j+1)*a_width] = a[i*num+j]
    return data

#计算采样时每个样本的有效像素占比(1-背景占比)
def compute_effective_ratio(sample):
    sample = np.array(sample)
    height, width = sample.shape
    count = 0
    for i in range(height):
        for j in range(width):
            if sample[i][j] != 0:
                count += 1
    return count/(height*width)

# 得到混淆矩阵
def compute_confusion_matrix(a, label, class_num):
    a = np.array(a)
    row, col = a.shape
    b = [[0 for col in range(class_num)] for row in range(class_num)]
    b = np.array(b)
    for j in range(row):
        for k in range(col):
            b[label[j, k], a[j, k]] += 1
    return b

#将标签转为0,1,2...k
#以前为0的(背景)保持为0
def convert(a, b):
    c = np.array(b)
    q = c.shape[0]
    total_num, a_height = a.shape#750 1024
    for i in range(total_num):
        for j in range(a_height):
            for z in range(q):
                if a[i][j] == b[z]:
                    a[i][j] = z+1
    return a

#计算结果中每一类的acc
def compute_acc_class(list_label, confusion_matrix):
    a = []
    s = np.array(confusion_matrix).shape[0]
    for i in range(s):
        a.append(confusion_matrix[i][i]/list_label[i+1])
    return a

#计算iou
def compute_iou(confusion_matrix):
    iou = []
    FN = 0
    FP = 0
    confusion_matrix = np.array(confusion_matrix)
    s = confusion_matrix.shape[0]
    for i in range(s):
        TP = confusion_matrix[i, i]
        for j in range(s):
            if j == i:
                pass
            else:
                FN += confusion_matrix[i, j]
                FP += confusion_matrix[j, i]
        iou.append(TP/(FN+FP+TP))
        FN = 0
        FP = 0
    return iou


# def main():
#
#
# if __name__ == '__main__':
#     main()


