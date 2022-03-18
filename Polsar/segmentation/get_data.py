args = parser.parse_known_args()[0]
feature_names=next(os.walk(args.feature_folder))[2]
print(feature_names)

feature = []
for i in range(len(feature_names)):
    feature.append(cv2.cvtColor(mpimg.imread(args.feature_folder + "/%s" % feature_names[i]), cv2.COLOR_RGB2GRAY))
data = np.array(feature)#(23/124, 750, 1024)
print(data.shape)

data = data.transpose((1, 2, 0))#(750, 1024, 23)
print(data.shape)
data = data/255
data.dtype#dtype('float64')

data_train = data
data_test = copy.deepcopy(data)#用于生成结果
