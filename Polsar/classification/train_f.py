import os
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
import numpy as np
from get_data import get_data
from get_patch_dataset import get_patch_dataset
from resnet_cbam import ResNet
from get_train_val_test import get_dataset
from data_preprocess import data_prepro

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

    classes = ('bare soil', 'barley', 'beet', 'buildings', 'forest', 'grass', 'lucerne', 'peas', 'potatoes'
               , 'rapseed', 'stembeans', 'water', 'wheat', 'wheat2', 'wheat3')
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
    torch.backends.cudnn.benchmark = False  #不要先也可以

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_folder', type=str, default='./features_23')
    parser.add_argument('--Short_side', type=int, default=750)
    parser.add_argument('--f_num', type=int, default=23)
    parser.add_argument('--patch_size', type=int, default=9)
    parser.add_argument('--label_folder', type=str, default=r'./label_f1.png')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--class_num', type=int, default=16)
    parser.add_argument('--train_ratio', type=float, default=0.01)#训练集和验证集取50%，测试集50%
    parser.add_argument('--test_ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--out_dir', type=str, default='./output/')
    args = parser.parse_known_args()[0]

    fix_all_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #获取数据和标签
    data, label = get_data(args)

    # 获取patch形式数据集
    all_sample, all_sample_l, labeled_sample, labeled_sample_l = get_patch_dataset(args, data, label)

    #分层抽样 获取训练/验证/测试集
    data_train, label_train, data_val, label_val, data_test, label_test = get_dataset(args, label, labeled_sample, labeled_sample_l)

    #简单预处理 归一化到-1，1 维度变换和类型变换
    data_train, label_train, data_val, label_val, data_test, label_test = data_prepro(data_train, label_train, data_val, label_val, data_test, label_test)

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