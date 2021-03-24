import argparse
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import GT_Prior_MSD, get_train_val_loader
from train_diff_models import train_UNet, train_MOUNet, train_NFTNet, train_SubNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

import wandb
# wandb login --relogin

# wandb.init(project="test_all", entity="rahuljr")

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--n_epochs_phase1", type=int, default=50, help="number of epochs of training")
parser.add_argument("--n_epochs_phase2", type=int, default=50, help="number of epochs of training")
parser.add_argument("--wd", type=float, default=0, help="weight decay")
# parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--data_file", type=str, default="../Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
parser.add_argument("--train_ratio", type=float, default=1.0, help="Number less than or equal to 1.0")
parser.add_argument("--model_type", type=str, default='unet', help="unet, mounet, nftnet, subnet")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for segmentation")
parser.add_argument("--n_classes_phase1", type=int, default=2, help="number of classes for segmentation")
parser.add_argument("--n_classes_phase2", type=int, default=3, help="number of classes for segmentation")
parser.add_argument("--prior_channel_bool", type=int, default=1.0, help="number of classes for segmentation")
parser.add_argument("--n_channels", type=int, default=2, help="number of classes for segmentation")
# parser.add_argument("--img_height", type=int, default=256, help="size of image height")
# parser.add_argument("--img_width", type=int, default=256, help="size of image width")
# parser.add_argument("--sample_interval", type=int, default=1, help="epochs after which we sample of images from generators")
# parser.add_argument("--checkpoint_interval", type=int, default=200, help="epochs between model checkpoints")
opt = parser.parse_args()
# print(opt)

if opt.use_gpu and torch.cuda.is_available():
	device = torch.device("cuda:1")
	print("Running on the GPU")
else:
	device = torch.device("cpu")
	print("Running on the CPU")

# Defining the dataloader for training and validation
root_dir = '../data/Task04_Hippocampus_processed/train/'
imgdir = 'imagesTr'
labeldir = 'labelsTr'
labeldir_left = 'labels_left'
labeldir_right = 'labels_right'
prior_list = os.listdir('../prior_models/best_model/')

input_type = 'img'
output_type = 'seg'

validation_split = 0.2
shuffle_dataset = True
random_seed= 42
train_ratio = opt.train_ratio

dataset = GT_Prior_MSD(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, \
	labeldir_right = labeldir_right, prior_list = prior_list, \
		device = device, train_ratio = train_ratio)

print('Loading training dataset is done')

# train_loader, validation_loader, weights = get_train_val_loader(dataset_obj = dataset, validation_split = validation_split, \
# 	batch_size = opt.batch_size, train_ratio = opt.train_ratio, n_cpus = opt.n_cpu)

# i, batch = next(enumerate(train_loader))
# print(batch['img'].shape)
# print(batch['cat_full'].shape)

# # Defining the dataloader for testing
# root_dir = '../data/Task04_Hippocampus_processed/test/'
# imgdir = 'imagesTest'
# labeldir = 'labelsTest'
# labeldir_left = 'labels_left'
# labeldir_right = 'labels_right'
# prior_list = os.listdir('../prior_models/best_model/')

# input_type = 'img'

# shuffle_dataset = True
# random_seed= 42

# test_dataset = Prior_MSD(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, \
# 	labeldir_right = labeldir_right, prior_list = prior_list, \
# 		device = device, train_ratio = 1)

# test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, num_workers=n_cpus)

# print('Loading test dataset is done')
# for tr in [1]:
for tr in [0.1, 1, 0.8, 0.6, 0.4, 0.2]:
	opt.train_ratio = tr
	train_loader, validation_loader, weights = get_train_val_loader(dataset_obj = dataset, validation_split = validation_split, \
		batch_size = opt.batch_size, train_ratio = opt.train_ratio, n_cpus = opt.n_cpu)


	print('Training UNet')
	opt.model_type = 'UNet'
	model_name = opt.model_type + '_' + str(opt.n_epochs) + '_' + str(opt.batch_size) + '_' + str(opt.lr) + \
		'_' + str(opt.train_ratio) + '_' + str(opt.n_channels) + '_' + str(opt.n_classes)
	wandb.init(project='test_all_GTSP_noweights', 
				name=model_name,
				reinit=True,
				config = opt
	)

	train_UNet(input_type, 'seg', wandb, train_loader, validation_loader, weights.cuda(), opt, device, model_name = model_name)


	print('Training MOUNet')
	opt.model_type = 'MOUNet'
	model_name = opt.model_type + '_' + str(opt.n_epochs_phase1) + '_' + str(opt.n_epochs_phase2) + '_' + str(opt.batch_size) + '_' + str(opt.lr) + \
		'_' + str(opt.train_ratio) + '_' + str(opt.n_channels) + '_' + str(opt.n_classes)
	wandb.init(project='test_all_GTSP_noweights', 
				name=model_name,
				reinit=True,
				config = opt
	)

	train_MOUNet(input_type, 'seg_left', 'seg', wandb, train_loader, validation_loader, weights.cuda(), opt, device, model_name = model_name)


	print('Training NFNet')
	opt.model_type = 'NFTNet'
	model_name = opt.model_type + '_' + str(opt.n_epochs_phase1) + '_' + str(opt.n_epochs_phase2) + '_' + str(opt.batch_size) + '_' + str(opt.lr) + \
		'_' + str(opt.train_ratio) + '_' + str(opt.n_channels) + '_' + str(opt.n_classes)
	wandb.init(project='test_all_GTSP_noweights', 
				name=model_name,
				reinit=True,
				config = opt
	)
	train_NFTNet(input_type, 'seg_left', 'seg_right', 'seg', wandb, train_loader, validation_loader, weights.cuda(), opt, device, model_name = model_name)

	print('Training SUBNet')
	opt.model_type = 'SUBNet'
	model_name = opt.model_type + '_' + str(opt.n_epochs_phase1) + '_' + str(opt.n_epochs_phase2) + '_' + str(opt.batch_size) + '_' + str(opt.lr) + \
		'_' + str(opt.train_ratio) + '_' + str(opt.n_channels) + '_' + str(opt.n_classes)
	wandb.init(project='test_all_GTSP_noweights', 
				name=model_name,
				reinit=True,
				config = opt
	)
	train_SubNet(input_type, 'seg_left', 'seg_full', wandb, train_loader, validation_loader, weights.cuda(), opt, device, model_name = model_name)
