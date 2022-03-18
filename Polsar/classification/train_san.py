import os
import cv2
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time
from tqdm import tqdm
import argparse
import numpy as np
import matplotlib.image as mpimg
from collections import Counter
from resnet import ResNet

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

def train_one_epoch(epoch_index, training_loader, device, optimizer, model, loss_fn):
    running_loss = 0. #统计n次迭代的loss
    last_loss = 0.#最后一次的running_loss

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 300 == 299:
            last_loss = running_loss / 300
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    return last_loss

def validation(model, device, validation_loader, loss_fn):
    # We don't need gradients on to do reporting
    model.train(False)
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    return avg_vloss

def test(args, model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    class_correct = list(0. for i in range(args.class_num - 1))
    class_total = list(0. for i in range(args.class_num - 1))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            # 获得每类acc
            c = (predicted == target).squeeze()
            for i in range(args.batch_size - 1):
                if len(target) != args.batch_size:
                    break
                label = target[i]
                class_correct[label - 1] += c[i].item()
                class_total[label - 1] += 1

    classes = ('low density urban', 'high density urban', 'vegetation', 'vegetation', 'ocean')
    print('Accuracy of the network on the test images: %9f %%' % (100 * correct / total))
    for i in range(args.class_num - 1):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 不要先也可以

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, default='./feature_san_124')
    parser.add_argument('--f_num', type=int, default=124)
    parser.add_argument('--patch_size', type=int, default=9)
    parser.add_argument('--label_folder', type=str, default=r'./label_s.png')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--class_num', type=int, default=6)
    parser.add_argument('--train_ratio', type=float, default=0.01)#训练集和验证集取50%，测试集50%
    parser.add_argument('--test_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='./output/')
    args = parser.parse_known_args()[0]

    fix_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #获取数据和标签
    feature_names = next(os.walk(args.feature_folder))[2]
    feature = []
    for i in range(len(feature_names)):
        feature.append(cv2.cvtColor(mpimg.imread(args.feature_folder + "/%s" % feature_names[i]), cv2.COLOR_RGB2GRAY))
    data = np.array(feature).transpose((1, 2, 0))
    label = cv2.imread(args.label_folder, 0)
    label = convert(label, np.unique(label)[1:])  # 生成（0，15）的标签 size=（750，1024）

    # 镜像padding
    image = data.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = torch.from_numpy(image)
    image = image.float()

    p = int((args.patch_size - 1) / 2)
    pad_image = torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')
    pad_image = pad_image.squeeze(0).permute(1, 2, 0).numpy()

    # 获取patch形式数据集
    all_sample = []
    all_sample_l = []
    labeled_sample = []
    labeled_sample_l = []
    for i in tqdm(range(900)):
        i += p
        for j in range(1024):
            j += p
            all_sample.append(pad_image[i - p:i + p + 1, j - p:j + p + 1, :])
            all_sample_l.append(label[i - p, j - p])
            if label[i - p, j - p] != 0:
                labeled_sample.append(pad_image[i - p:i + p + 1, j - p:j + p + 1, :])
                labeled_sample_l.append(label[i - p, j - p])
    all_sample = np.array(all_sample)
    all_sample_l = np.array(all_sample_l)
    labeled_sample = np.array(labeled_sample)
    labeled_sample_l = np.array(labeled_sample_l)

    # 获取每一类应取的数目
    count = Counter(label.reshape(900 * 1024))
    class_train_num = []
    for i in range(1, args.class_num):
        class_train_num.append(int(count[i] * args.train_ratio))

    # 每一类的数据放入字典中 方便取10%
    # 初始化字典
    class_sample = {}
    class_label = {}
    for i in range(1, args.class_num):
        class_sample[i] = []
        class_label[i] = []
    for k in range(1, args.class_num):
        for i in range(len(labeled_sample)):
            if labeled_sample_l[i] == k:
                class_sample[k].append(labeled_sample[i, :])
                class_label[k].append(k)

    # 获取训练集 每一类取10%
    data_train = []
    label_train = []
    data_val = []
    label_val = []
    data_test = []
    label_test = []
    for i in tqdm(range(1, args.class_num)):
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
    # 归一化到-1，1
    data_train = (torch.from_numpy(data_train) / 255) * 2 - 1
    label_train = torch.from_numpy(label_train)
    data_val = (torch.from_numpy(data_val) / 255) * 2 - 1
    label_val = torch.from_numpy(label_val)
    data_test = (torch.from_numpy(data_test) / 255) * 2 - 1
    label_test = torch.from_numpy(label_test)

    data_train = data_train.permute(0, 3, 1, 2)
    data_val = data_val.permute(0, 3, 1, 2)
    data_test = data_test.permute(0, 3, 1, 2)
    label_train = label_train.long()
    label_val = label_val.long()
    label_test = label_test.long()

    dataset_train = TensorDataset(data_train, label_train)
    dataset_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataset_val = TensorDataset(data_val, label_val)
    dataset_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    dataset_test = TensorDataset(data_test, label_test)
    dataset_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    model = ResNet(num_class=args.class_num, f_num=args.f_num).to(device)
    loss_fn = nn.CrossEntropyLoss()#不用移动到cuda
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(args.epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, dataset_train, device, optimizer, model, loss_fn)
        avg_vloss = validation(model, device, dataset_val, loss_fn)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            # model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            # model_path = args.out_dir + 'model_best'
            model_path = 'model_best'
            print('...............................save best model at {} epoch'.format(epoch_number + 1))
            torch.save(model.state_dict(), model_path)
        epoch_number += 1

    # inference
    PATH = model_path
    saved_model = ResNet(num_class=args.class_num, f_num=args.f_num).to(device)
    saved_model.load_state_dict(torch.load(PATH))

    test(args, model, device, dataset_test)


if __name__ == '__main__':
    main()