import random

def sample_patch(args, data_train):
    if args.seed:
        random.seed(args.seed)
    count = 0
    sample_data = []
    sample_label = []
    X_h, X_w, _ = data_train.shape
    #按步长采样
    if args.Sampling_by_step:
        for i in range(0, X_h - args.img_h - 1, args.img_h):
            for j in range(0, X_w - args.img_w - 1, args.img_w):
                data_sample = data_train[i: i + args.img_h, j: j + args.img_w, :]
                label_sample = label[i: i + args.img_h, j: j + args.img_w]
                r = compute_effective_ratio(label_sample)
                if r > args.effective_ratio:
                    sample_data.append(data_sample)
                    sample_label.append(label_sample)
    #随机采样
    while count < args.random_sample_num: 
        random_height = random.randint(0, X_h - args.img_h - 1)
        random_width = random.randint(0, X_w - args.img_w - 1)
        data_sample = data_train[random_height: random_height + args.img_h, random_width: random_width + args.img_w, :]
        label_sample = label[random_height: random_height + args.img_h, random_width: random_width + args.img_w]
        r = compute_effective_ratio(label_sample)
        if r > args.effective_ratio:
          sample_data.append(data_sample)
          sample_label.append(label_sample)
        count += 1
    length=len(sample_data)

    if args.sample_buildings：
        count = 0
        while count < args.random_sample_num_building: 
            random_height = random.randint(667, 667+args.img_h)
            random_width = random.randint(667, 667+args.img_w)
            data_sample = data_train[random_height: random_height + args.img_h, random_width: random_width + args.img_w, :]
            label_sample = label[random_height: random_height + args.img_h, random_width: random_width + args.img_w]
            r = compute_effective_ratio(label_sample)
            if r > args.effective_ratio_building:
                sample_data.append(data_sample)
                sample_label.append(label_sample)
            count += 1
        print("Collected {} building samples".format(len(sample_data)-length))

    train_test_split = int(np.array(sample_data).shape[0] * args.train_ratio)

    train_dataset = sample_data[0:train_test_split]
    train_label = sample_label[0:train_test_split]
    test_dataset = sample_data[train_test_split:]
    test_label = sample_label[train_test_split:]
    print("training set size {}".format(np.array(train_dataset).shape))
    print("training label size {}".format(np.array(train_label).shape))
    print("test set size {}".format(np.array(test_dataset).shape))
    print("test label size {}".format(np.array(test_label).shape))
    return train_dataset,train_label,test_dataset,test_label
