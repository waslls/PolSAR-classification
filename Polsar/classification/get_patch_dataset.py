import torch
import numpy as np
from tqdm import tqdm

# 镜像padding
def mirror_padding(args, data):
    image = data.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    image = torch.from_numpy(image)
    image = image.float()

    p = int((args.patch_size - 1) / 2)
    pad_image = torch.nn.functional.pad(image, (p, p, p, p), mode='reflect')
    pad_image = pad_image.squeeze(0).permute(1, 2, 0).numpy()
    return pad_image

# 获取patch形式数据集
def get_patch_dataset(args, data, label):
    pad_image = mirror_padding(args, data)

    all_sample = []
    all_sample_l = []
    labeled_sample = []
    labeled_sample_l = []

    p = int((args.patch_size - 1) / 2)
    for i in tqdm(range(args.Short_side)):
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
    return all_sample, all_sample_l, labeled_sample, labeled_sample_l