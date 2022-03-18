import os
import cv2
import copy
import random
import argparse
import numpy as np
import tensorflow as tf
from unet import create_model
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras import layers, optimizers, datasets, Sequential
from get_data import get_data
from sample_patch import sample_patch
from data_aug import data_aug
from focal_loss import focal_loss_softmax
from sklearn.metrics import confusion_matrix

def train(dataset_train, model, creterion, optimizer):
    for idx, (data, label) in enumerate(dataset_train):
        with tf.GradientTape() as tape:
            y_perd = model(data)
			if args.focal:
				loss = focal_loss_softmax(label, y_perd, args.num_class, alpha)
			else:
            	loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label, y_pred=y_perd, from_logits=True)
        grads = tape.gradient(loss, conv_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, conv_net.trainable_variables))
        loss_metrics(loss)   
        acc_metrics(label, y_perd)
    return loss_metrics.result(), acc_metrics.result()
            
def test(dataset_test, model, creterion):
    for idx, (data, label) in enumerate(dataset_test):
        y_perd = model(data)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=label, y_pred=y_perd, from_logits=True)
        loss_metrics(loss)   
        acc_metrics(label, y_perd)
    return loss_metrics.result(), acc_metrics.result()

#alpha for focal loss
# alpha=[1.]*args.num_class
a=Counter(label.flatten())
w = list(1/([a[i] for i in range(16)]/sum(sum(label!=0))))
w = w/sum(w)
alpha=w.astype(np.float32)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--net', type=str, default='UNet')
	parser.add_argument('--img_w', type=int, default=32)
	parser.add_argument('--img_h', type=int, default=32)
	parser.add_argument('--lr', type=float, default=1e-4)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=8)
	parser.add_argument('--feature_folder', type=str, default='./features_fluolida')#'./features_124'、'./features_fluolida'
	parser.add_argument('--label_folder', type=str, default=r'./label_f1.png')
	parser.add_argument('--save_freq', type=int, default=2)
	parser.add_argument('--checkpoint_folder', type=str, default='./output')
	parser.add_argument('--data_aug', type=int, default=1)
	parser.add_argument('--train_ratio', type=float, default=0.8)
	parser.add_argument('--effective_ratio', type=float, default=0.65)
	parser.add_argument('--effective_ratio_building', type=float, default=0.35)
	parser.add_argument('--buffer_size', type=float, default=100)
	parser.add_argument('--num_class', type=int, default=16)
	parser.add_argument('--seed', type=int, default=2021)
	parser.add_argument('--lr_decay', type=int, default=0)
	parser.add_argument('--save_train', type=int, default=0)
	parser.add_argument('--val_acc', type=int, default=0)
	parser.add_argument('--focal', type=int, default=0)
	parser.add_argument('--Sampling_by_step', type=int, default=1)
	parser.add_argument('--sample_buildings', type=int, default=1)
	parser.add_argument('--random_sample_num', type=int, default=1000)
	parser.add_argument('--random_sample_num_building', type=int, default=100)
	parser.add_argument('--user_defined', type=int, default=1)
	args = parser.parse_known_args()[0]
	
	data_train, data_all, label = get_data(args)
	train_dataset,train_label,test_dataset,test_label = sample_patch(args,data_train,label)
	train_dataset, train_label = data_aug(train_dataset,train_label)
	
	train_dataset = tf.cast(train_dataset, tf.float32)*2-1
	test_dataset = tf.cast(test_dataset, tf.float32)*2-1
	train_label = np.expand_dims(train_label, axis=3)
	test_label = np.expand_dims(test_label, axis=3)
	
	dataset_train = tf.data.Dataset.from_tensor_slices((train_dataset, train_label))#((32, 32, 23), (32, 32, 1)), types: (tf.float32, tf.uint8)>
	dataset_test = tf.data.Dataset.from_tensor_slices((test_dataset, test_label))#((32, 32, 23), (32, 32, 1)), types: (tf.float32, tf.uint8)>
	dataset_train = dataset_train.shuffle(buffer_size=args.buffer_size).batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)##<PrefetchDataset shapes: ((None, 32, 32, 23), (None, 32, 32, 1)), types: (tf.float32, tf.uint8)>
	dataset_test = dataset_test.batch(args.batch_size)##<BatchDataset shapes: ((None, 32, 32, 23), (None, 32, 32, 1)), types: (tf.float32, tf.uint8)>
	
	if args.user_defined==0:
		class MeanIoU(tf.keras.metrics.MeanIoU):  # 针对非独热编码
			def __call__(self, y_true, y_pred, sample_weight=None):
				y_pred = tf.argmax(y_pred, axis=-1)
				return super().__call__(y_true, y_pred, sample_weight=sample_weight)
		conv_net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
						loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
						metrics=['sparse_categorical_accuracy', MeanIoU(num_classes=16)])
	else:
		acc_sum = 0
		acc_mean_class = [0 for i in range(args.num_class-1)]
		for g in range(10):
			conv_net = create_model(args.img_w, args.img_h, len(feature_names), args.num_class)
			creterion = tf.keras.losses.SparseCategoricalCrossentropy()
			optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
			loss_metrics = tf.keras.metrics.Mean(name='loss')
			acc_metrics = tf.keras.metrics.SparseCategoricalAccuracy('acc')
			for epoch in range(args.num_epochs):
				train_loss, train_acc = train(dataset_train=dataset_train, 
								  model=conv_net, 
								  creterion=creterion, 
								  optimizer=optimizer)
				loss_metrics.reset_states()
				acc_metrics.reset_states()
				if args.lr_decay and train_acc>0.975:
					optimizer.learning_rate = optimizer.learning_rate*0.98

				print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f}")
				if args.save_train and epoch > 15 and (epoch % args.save_freq == 0 or epoch == args.num_epochs):#仅仅借助train loss 或 acc保存。
					model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}-Acc-{train_acc}")
					conv_net.save_weights(model_path)
					print(f'----- Save model: {model_path}.tf_params')


				test_loss, test_acc = test(dataset_test=dataset_test, 
								  model=conv_net, 
								  creterion=creterion)
				loss_metrics.reset_states()
				acc_metrics.reset_states()
				print(f"----- Epoch[{epoch}/{args.num_epochs}] Test Loss: {test_loss:.4f} Test Acc: {test_acc:.4f}")
				if args.val_acc:#借助train和test acc来保存模型和终止训练，未用loss数据。
					if test_acc>0.98 and signal == 0:
						model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Acc-{test_acc}-Seed-{args.seed}")
						conv_net.save_weights(model_path)
						print(f'----- Save model: {model_path}.tf_params')
						signal = 1
					if test_acc>0.99 and train_acc>0.99:
						model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Acc-{test_acc}-Seed-{args.seed}")
						conv_net.save_weights(model_path)
						print(f'----- Save model: {model_path}.tf_params')
						break

			model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Loop-{g}-Epoch-{epoch}-tLoss-{train_loss}-tAcc-{train_acc}-eLoss-{test_loss}-eAcc-{test_acc}")
			conv_net.save_weights(model_path)
			print(f'----- Save model: {model_path}.tf_params')

			result = conv_net.predict(data_test)#(1024, 32, 32, 16)
			result = tf.argmax(result, axis=-1)
			result = combination(result)#(1024, 1024)
			plt.imshow(result)#[ 0  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
			plt.show()
			np.savetxt('res'+str(g),result)
			np.unique(result)

			result = result[0:750]
			result[label==0] = 0

			confusion_matrix=compute_confusion_matrix(result,label,args.num_class)[1:,1:]
			count_converted = Counter(label.flatten())#返回的是字典 key是元素的值 value是这个元素的个数
			#每类准确率
			acc_class = compute_acc_class(count_converted, confusion_matrix)
			print(acc_class)
			for i in range(args.num_class-1):
				acc_mean_class[i] += 100 * acc_class[i]
				print(acc_mean_class[i]/(g+1))
			#总体准确率
			acc = np.trace(confusion_matrix)/np.array([label!=0]).sum()
			acc_sum += 100 * acc
			print('Accuracy of the network on the test images: %9f %%' % (100*acc))
			print('acc_mean of %d loop is %5f' % (g+1, acc_sum / (g + 1)))
	
	
if __name__ == '__main__':
    main()
  
