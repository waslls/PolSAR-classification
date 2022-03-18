import numpy as np
from augment import data_augments_brightsness, data_augments_contrast, data_label_augments_lr_flip, data_augments_saturation, data_label_augments_updown_flip, data_augment_add_noise, data_augment_rotate

def data_aug(args, train_dataset,train_label):
    if args.data_aug:
        print('start augmentation')
        train_dataset1 = train_dataset
        train_label1 = train_label

        rotate_data, rotate_label = data_augment_rotate(train_dataset1, train_label1)
        train_dataset = np.concatenate([train_dataset, rotate_data], axis=0)
        train_label = np.concatenate([train_label, rotate_label], axis=0)
        print(np.array(train_dataset).shape)
        print(np.array(train_label).shape)

        noise_data, noise_label = data_augment_add_noise(train_dataset1, train_label1)
        train_dataset = np.concatenate([train_dataset, noise_data], axis=0)
        train_label = np.concatenate([train_label, noise_label], axis=0)
        print(np.array(train_dataset).shape)
        print(np.array(train_label).shape)

        lr_flip_data, lr_flip_label = data_label_augments_lr_flip(train_dataset1, train_label1)
        train_dataset = np.concatenate([train_dataset, lr_flip_data], axis=0)
        train_label = np.concatenate([train_label, lr_flip_label], axis=0)
        print(np.array(train_dataset).shape)
        print(np.array(train_label).shape)

        updown_flip_data, updown_flip_label = data_label_augments_updown_flip(train_dataset1, train_label1)
        train_dataset = np.concatenate([train_dataset, updown_flip_data], axis=0)
        train_label = np.concatenate([train_label, updown_flip_label], axis=0)
        print(np.array(train_dataset).shape)
        print(np.array(train_label).shape)

        return train_dataset, train_label
