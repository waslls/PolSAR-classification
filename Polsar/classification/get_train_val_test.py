from collections import Counter
import numpy as np

# 获取每一类应取的数目
def get_cls_num(args, label, labeled_sample, labeled_sample_l):
    count = Counter(label.reshape(750 * 1024))
    class_train_num = []
    for i in range(1, args.class_num):
        class_train_num.append(int(count[i] * args.train_ratio))

    # 每一类的数据放入字典中 方便取x%
    class_sample = {}
    class_label = {}
    for i in range(1, args.class_num):
        class_sample[i] = []
        class_label[i] = []
    for k in range(1, 16):
        for i in range(len(labeled_sample)):
            if labeled_sample_l[i] == k:
                class_sample[k].append(labeled_sample[i, :])
                class_label[k].append(k)
    return class_sample, class_label

# 获取训练集 每一类取x%
def get_dataset(args, label, labeled_sample, labeled_sample_l):
    class_sample, class_label = get_cls_num(args, label, labeled_sample, labeled_sample_l)
    data_train = []
    label_train = []
    data_val = []
    label_val = []
    data_test = []
    label_test = []
    for i in range(1, args.class_num):
        index = np.arange(len(class_sample[i]))
        np.random.shuffle(index)

        train_range = int(len(class_sample[i]) * args.train_ratio)
        val_range = train_range + int(len(class_sample[i]) * (1-args.test_ratio))

        data_train += list(
            np.array(class_sample[i])[index][:train_range, :])
        label_train += list(np.array(class_label[i])[index][:train_range])

        data_val += list(
            np.array(class_sample[i])[index][train_range:val_range, :])
        label_val += list(np.array(class_label[i])[index][train_range:val_range])

        data_test += list(
            np.array(class_sample[i])[index][val_range:, :])
        label_test += list(np.array(class_label[i])[index][val_range:])

    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_val = np.array(data_val)
    label_val = np.array(label_val)
    data_test = np.array(data_test)
    label_test = np.array(label_test)

    print(data_train.shape)
    print(label_train.shape)
    print(data_val.shape)
    print(label_val.shape)
    print(data_test.shape)
    print(label_test.shape)

    return data_train, label_train, data_val, label_val, data_test, label_test