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
	parser.add_argument('--Sampling_by_step', type=int, default=1)
	parser.add_argument('--sample_buildings', type=int, default=1)
	parser.add_argument('--random_sample_num', type=int, default=1000)
	parser.add_argument('--random_sample_num_building', type=int, default=100)
	parser.add_argument('--user_defined', type=int, default=1)
	args = parser.parse_known_args()[0]
	
	
	
	
	
	
	
if __name__ == '__main__':
    main()
  
