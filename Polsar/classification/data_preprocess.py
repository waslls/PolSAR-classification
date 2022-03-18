import torch

# 归一化到-1，1
def data_prepro(data_train, label_train, data_val, label_val, data_test, label_test):
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
    return data_train, label_train, data_val, label_val, data_test, label_test