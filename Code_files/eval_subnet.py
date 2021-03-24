import argparse
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import Prior_MSD, get_train_val_loader, Mean_Prior_MSD, GT_Prior_MSD
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
parser.add_argument("--n_classes_phase2", type=int, default=2, help="number of classes for segmentation")
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

dataset = Mean_Prior_MSD(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, \
	labeldir_right = labeldir_right, prior_list = prior_list, \
		device = device, train_ratio = train_ratio)

print('Loading training dataset is done')

train_loader, validation_loader, weights = get_train_val_loader(dataset_obj = dataset, validation_split = validation_split, \
    batch_size = 1, train_ratio = opt.train_ratio, n_cpus = opt.n_cpu)


model_left = UNet2D(opt.n_classes_phase1, opt.n_channels).cuda()
model_full = UNet2D(opt.n_classes_phase1, opt.n_channels).cuda()

model_dir = '../models_mean_SP_noweights/SUBNet/'

files = os.listdir(model_dir)
for j in files:
    if 'left' in j:
        files.remove(j)

full_files = sorted(files)

df = []
model_full_name = model_dir + 'SUBNet_50_50_4_0.0002_1_2_3_full__epoch_50.pt'
model_left_name = model_dir + 'SUBNet_50_50_4_0.0002_1_2_3_left__epoch_50.pt'
# for model_full_name in files:
df_row = []
model_left_name = model_full_name.replace('full', 'left')
model_full.load_state_dict(torch.load(model_full_name))
model_left.load_state_dict(torch.load(model_left_name))

dice_right = eval_subnet_right(model_full, model_left, 2, validation_loader, 'img', 'seg_left', 'seg_right', 'seg_full', wandb)

df_row.append(model_full_name)
df_row.append(model_left_name)
df_row.append(dice_right)

df.append(df_row)

print(df)