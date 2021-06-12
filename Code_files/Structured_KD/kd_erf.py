import argparse
import logging
import os
import pdb
from torch.autograd import Variable
import os.path as osp
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import resource
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loss import cross_ent_dice_loss, pixel_wise_KL, pair_wise_loss, dice_coeff_multiclass, Cross_ent_dist
from utils import print_model_parm_nums
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as T

from erfnet import Net
from student import ENet
from teacher import UNet2D
from sagan_models import Discriminator

def crop(tensor,target_tensor): # Crop tensor to target tensor size
	# print(target_tensor.shape)
	target_shape1, target_shape2 = target_tensor.shape[1], target_tensor.shape[2]
	return T.CenterCrop((target_shape1, target_shape2))(tensor)

class KD_model():
	def __init__(self, args, lambda1, lambda2):

		cudnn.enabled = True
		self.args = args
		# device = torch.device("cuda:0")
		self.lambda1 = lambda1
		self.lambda2 = lambda2

		# Student Model
		# student = ENet(C=1, out_c=3).cuda()
		# student = UNet2D(num_classes=3, num_channels=1).cuda()
		student = Net(1,3).cuda()
		print_model_parm_nums(student, 'student_model')
		self.student = student

		# Teacher Model
		teacher = UNet2D(num_classes=3, num_channels=1).cuda()
		teacher.load_state_dict(torch.load('/home/rahulroy/Segmentation_DA_MSD/models_noSP/UNet/UNet_100_4_0.0002_0.1_1_3_epoch_100.pt'))
		print_model_parm_nums(teacher, 'teacher_model')
		self.teacher = teacher

		# Discriminator
		D_model = Discriminator(args.preprocess_GAN_mode, args.n_classes, args.batch_size, 64, 64).cuda()
		print_model_parm_nums(D_model, 'D_model')
		self.D_model = D_model

		self.G_solver = torch.optim.Adam(self.student.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.G_solver, factor=0.5, patience=2, min_lr=1e-6)
		self.D_solver = torch.optim.Adam(self.D_model.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
		# self.G_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, self.student.parameters()), 'initial_lr': args.lr_g}], args.lr_g, momentum=args.momentum, weight_decay=args.weight_decay)
		# self.D_solver = optim.SGD([{'params': filter(lambda p: p.requires_grad, D_model.parameters()), 'initial_lr': args.lr_d}], args.lr_d, momentum=args.momentum, weight_decay=args.weight_decay)

		self.cross_entropy_dice = cross_ent_dice_loss #CriterionCrossEntropy()
		self.pixel_wise_KL = pixel_wise_KL
		# self.pixel_wise_KL = Cross_ent_dist
		self.pair_wise_loss = pair_wise_loss
		self.softmax = nn.Softmax(dim=1)
		# self.criterion_adv = CriterionAdv(args.adv_loss_type)
		# if args.adv_loss_type == 'wgan-gp':
		#     self.criterion_AdditionalGP = CriterionAdditionalGP(self.parallel_D, args.lambda_gp)
		# self.criterion_adv_for_G = CriterionAdvForG(args.adv_loss_type)
			
		self.ce_G_loss = 0.0
		self.pi_G_loss = 0.0
		self.pa_G_loss = 0.0
		# self.D_loss = 0.0

		# cudnn.benchmark = True
		# if not os.path.exists(args.snapshot_dir):
		#     os.makedirs(args.snapshot_dir)
	
	def train_loop(self, batch):
		self.G_solver.zero_grad()
		input_img = Variable(batch['img']).cuda()
		segs = Variable(batch['seg']).type(torch.LongTensor).cuda()
		with torch.no_grad():
			self.teacher.eval()
			self.preds_T = self.teacher(input_img)
		self.student.train()
		self.preds_S = self.student(input_img)
		self.preds_S[0] = crop(self.preds_S[0], segs)

		#Calculating dice coeff
		dscoeff, outs, seg_out = dice_coeff_multiclass(segs, self.preds_S[0], 3)

		G_loss = 0
		ce_loss = self.cross_entropy_dice(torch.squeeze(segs, dim=1), self.preds_S[0])
		self.ce_G_loss = ce_loss.item()
		G_loss += ce_loss

		if self.args.pi == True:
			pix_kl = self.pixel_wise_KL(self.softmax(self.preds_S[0]), self.softmax(self.preds_T[0]))
			self.pi_G_loss = pix_kl.item()
			G_loss += self.lambda1 * pix_kl

		if self.args.pa == True:
			pair_loss = 0
			for i in range(1,3):
				stud = F.upsample(self.preds_S[i], size = (self.preds_T[i].shape[2], self.preds_T[i].shape[3]),mode='bicubic' )
				pair_loss += pair_wise_loss(stud, self.preds_T[i])
			self.pa_G_loss = pair_loss.item()
			G_loss += self.lambda2 * pair_loss
		
		# G_loss = ce_loss #+ 0.1* pix_kl + 0.1 * pair_loss
		G_loss.backward()
		self.G_loss =  G_loss.item()
		# print(self.G_loss, dscoeff, sum(dscoeff)/(3), self.pi_G_loss, self.pa_G_loss)
		self.G_solver.step()

		return G_loss, dscoeff

	def validate_model(self, model, wandb, num_classes, validation_loader, table_val):

		val_losses = []
		val_dscoeffs = []
		val_avg_losses = []
		val_avg_dscoeffs = []

		val_avg_loss = 0.0
		val_avg_dscoeff = 0.0

		# model.eval()
		pbar_val = tqdm(total = len(validation_loader), desc = "Val: Avg loss{}, avg_dice{}"\
				.format(val_avg_loss,val_avg_dscoeff), leave=False)
		pbar_val.n = 0
		
		count = 0
		val_dscoeffs = []
		for i, batch in enumerate(validation_loader):
			with torch.no_grad():
				count+=1
				input_img = Variable(batch['img']).cuda()
				segs = Variable(batch['seg']).type(torch.LongTensor).cuda()
				outputs = self.student(input_img)
				outputs = crop(outputs[0], segs)
				
				loss = cross_ent_dice_loss(torch.squeeze(segs, dim=1), outputs)
				# loss.backward()
				# optimizer.step()
				dscoeff, outs, segs = dice_coeff_multiclass(segs, outputs, num_classes)
				
				val_dscoeffs.append(dscoeff)

				val_avg_loss += loss.item()
				val_avg_dscoeff += sum(dscoeff)/(num_classes)
				pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".format\
					(val_avg_loss/count,val_avg_dscoeff/count))
				pbar_val.update(1)
				
		val_dscoeffs.append(np.mean(np.array(val_dscoeffs), axis=0))
		val_avg_dscoeffs.append(val_avg_dscoeff/len(validation_loader))
		val_avg_losses.append(val_avg_loss/len(validation_loader))
		pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".\
			format(val_avg_loss/len(validation_loader),val_avg_dscoeff/len(validation_loader)))
		wandb.log({"Val Avg loss": round(val_avg_loss/len(validation_loader), 3), "Val Avg dice": round(val_avg_dscoeff/len(validation_loader), 3)})
		new = list(val_dscoeffs[-1])
		new.append(val_avg_dscoeff/len(validation_loader))
		table_val.append(new)

		return val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val



