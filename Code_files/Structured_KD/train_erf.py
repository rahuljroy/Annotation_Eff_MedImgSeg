import sys
import torch
import os
import argparse
import numpy as np
# from utils.train_options import TrainOptions
from kd_erf import KD_model
import pandas as pd
from torch.autograd import Variable
from loss import cross_ent_dice_loss, dice_coeff_multiclass
import logging
import warnings
from tqdm import tqdm
import wandb


from dataset import Mean_Prior_MSD, get_train_val_loader, Spine

# def validate_model(model, wandb, num_classes, validation_loader, table_val):

#     val_losses = []
#     val_dscoeffs = []
#     val_avg_losses = []
#     val_avg_dscoeffs = []

#     val_avg_loss = 0.0
#     val_avg_dscoeff = 0.0

#     model.eval()
#     pbar_val = tqdm(total = len(validation_loader), desc = "Val: Avg loss{}, avg_dice{}"\
#             .format(val_avg_loss,val_avg_dscoeff), leave=False)
#     pbar_val.n = 0
    
#     count = 0
#     val_dscoeffs = []
#     for i, batch in enumerate(validation_loader):
#         with torch.no_grad():
#             count+=1
#             input_img = Variable(batch['img']).cuda()
#             segs = Variable(batch['seg']).type(torch.LongTensor).cuda()
#             outputs = model(input_img)
            
#             loss = cross_ent_dice_loss(torch.squeeze(segs, dim=1), outputs[0])
#             # loss.backward()
#             # optimizer.step()
#             dscoeff, outs, segs = dice_coeff_multiclass(segs, outputs[0], num_classes)
            
#             val_dscoeffs.append(dscoeff)

#             val_avg_loss += loss.item()
#             val_avg_dscoeff += sum(dscoeff)/(num_classes)
#             pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".format\
#                 (val_avg_loss/count,val_avg_dscoeff/count))
#             pbar_val.update(1)
            
#     val_dscoeffs.append(np.mean(np.array(val_dscoeffs), axis=0))
#     val_avg_dscoeffs.append(val_avg_dscoeff/len(validation_loader))
#     val_avg_losses.append(val_avg_loss/len(validation_loader))
#     pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".\
#         format(val_avg_loss/len(validation_loader),val_avg_dscoeff/len(validation_loader)))
#     wandb.log({"Val Avg loss": round(val_avg_loss/len(validation_loader), 3), "Val Avg dice": round(val_avg_dscoeff/len(validation_loader), 3)})
#     new = list(val_dscoeffs[-1])
#     new.append(val_avg_dscoeff/len(validation_loader))
#     table_val.append(new)

#     return val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val

root_dir = '../../data/SpineCT_processed/train/'
imgdir = 'imagesTr'
labeldir = 'labelsTr'
labeldir_left = 'labels_left'
labeldir_right = 'labels_right'
# prior_list = os.listdir('../../prior_models/best_model/')

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20, help="number of epochs of training")
# parser.add_argument("--n_epochs_phase1", type=int, default=50, help="number of epochs of training")
# parser.add_argument("--n_epochs_phase2", type=int, default=50, help="number of epochs of training")
parser.add_argument("--wd", type=float, default=0, help="weight decay")
# parser.add_argument("--data_file", type=str, default="../Sample_Data_Readme_and_other_docs", help="location of dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--use_gpu", type=int, default=1, help="cpu: 0, gpu: 1")
parser.add_argument("--train_ratio", type=float, default=1.0, help="Number less than or equal to 1.0")
parser.add_argument("--model_type", type=str, default='enet', help="unet, mounet, nftnet, subnet")
parser.add_argument("--n_classes", type=int, default=3, help="number of classes for segmentation")
# parser.add_argument("--n_classes_phase", type=int, default=2, help="number of classes for segmentation")
# parser.add_argument("--prior_channel_bool", type=int, default=1.0, help="number of classes for segmentation")
parser.add_argument("--n_channels", type=int, default=1, help="number of classes for segmentation")
parser.add_argument("--preprocess-GAN-mode", type=int, default=1, help="preprocess-GAN-mode should be tanh or bn")
parser.add_argument("--pi", type=str, default='False', help="is pixel wise loss using or not")
parser.add_argument("--pa", type=str, default='False', help="is pair wise loss using or not")

args = parser.parse_args()


lambda1 = [1, 0.1, 0.01, 0.001, 0.0001]
lambda2 = [1, 0.1, 0.01, 0.001, 0.0001]
train_str = ['pi', 'pa']

validation_split = 0.2
shuffle_dataset = True
random_seed= 42
train_ratio = args.train_ratio

# dataset = Mean_Prior_MSD(root = root_dir, imgdir = imgdir, labeldir = labeldir, labeldir_left = labeldir_left, \
# 	labeldir_right = labeldir_right, prior_list = prior_list, \
# 		device = torch.device("cuda:1"), train_ratio = train_ratio)
dataset = Spine(root = root_dir, imgdir = imgdir, labeldir = labeldir, device = torch.device("cuda:1"))

train_loader, validation_loader = get_train_val_loader(dataset_obj = dataset, validation_split = validation_split, \
		batch_size = args.batch_size, train_ratio = args.train_ratio, n_cpus = args.n_cpu)

# for step, batch in enumerate(train_loader):
#     continue

args.pi = False
args.pa = False  
model = KD_model(args, 0, 0)
wandb.init(project='KD_ERFNet_spine', 
        name='ErfNet_ce_'+str(0)+'_'+str(0)+'_'+str(args.n_epochs),
        reinit=True,
        config = args
)
columns=["Background", "Anterior", "Posterior", "Avg Dice"]     

losses = []
dscoeffs_train = []
avg_losses = []
avg_dscoeffs = []

table_train = []
table_val = []

avg_loss = 0.0
avg_dscoeff = 0.0
val_avg_loss = 0.0
val_avg_dscoeff = 0.0

pbar_outer = tqdm(total=args.n_epochs, desc = 'Epoch count', leave=False)
pbar_train = tqdm(total = len(train_loader), desc = "Train: Avg loss{}, avg_dice{}"\
        .format(avg_loss,avg_dscoeff), leave=False)

for epoch in range(args.n_epochs):
    pbar_train.n = 0
    pbar_train.refresh()

    avg_loss = 0.0
    avg_dscoeff = 0.0
    
    count = 0
    dscoeffs = []

    for step, batch in enumerate(train_loader):
        count+=1
        loss, dscoeff = model.train_loop(batch)
        dscoeffs.append(dscoeff)
        avg_loss += loss.item()
        avg_dscoeff += sum(dscoeff)/(args.n_classes)
        pbar_train.set_description("Train: Avg lsoss: {:.3f}, Avg_dice: {:.3f}".\
            format(avg_loss/count,avg_dscoeff/count))
        pbar_train.update(1)
            
    dscoeffs_train.append(np.mean(np.array(dscoeffs), axis = 0))
    # print(dscoeffs_train)
    avg_losses.append(avg_loss/len(train_loader))
    avg_dscoeffs.append(avg_dscoeff/len(train_loader))
    pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}"\
        .format(avg_loss/len(train_loader),avg_dscoeff/len(train_loader)))
    wandb.log({"Train Avg loss": round(avg_loss/len(train_loader), 3), "Train Avg dice": round(avg_dscoeff/len(train_loader), 3)})
    new = list(dscoeffs_train[-1])
    new.append(avg_dscoeff/len(train_loader))
    table_train.append(new)
    model.scheduler.step(avg_loss/len(train_loader))

    val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
            model.validate_model(model.student, wandb, args.n_classes, validation_loader, table_val)
    pbar_outer.update(1)

table_train = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
table_val = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
wandb.log({"Train Table": table_train})
wandb.log({"Val Table": table_val})

for lamb1 in lambda1:
    args.pi = True
    args.pa = False
    model = KD_model(args, lamb1, 0)
    wandb.init(project='KD_ERFNet_spine', 
            name='ErfNet_pi_'+str(lamb1)+'_'+str(0)+'_'+str(args.n_epochs),
            reinit=True,
            config = args
    )

    columns=["Background", "Anterior", "Posterior", "Avg Dice"]     

    losses = []
    dscoeffs_train = []
    avg_losses = []
    avg_dscoeffs = []

    table_train = []
    table_val = []

    avg_loss = 0.0
    avg_dscoeff = 0.0
    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0

    pbar_outer = tqdm(total=args.n_epochs, desc = 'Epoch count', leave=False)
    pbar_train = tqdm(total = len(train_loader), desc = "Train: Avg loss{}, avg_dice{}"\
            .format(avg_loss,avg_dscoeff), leave=False)

    for epoch in range(args.n_epochs):
        pbar_train.n = 0
        pbar_train.refresh()

        avg_loss = 0.0
        avg_dscoeff = 0.0
        
        count = 0
        dscoeffs = []

        for step, batch in enumerate(train_loader):
            count+=1
            loss, dscoeff = model.train_loop(batch)
            dscoeffs.append(dscoeff)
            avg_loss += loss.item()
            avg_dscoeff += sum(dscoeff)/(args.n_classes)
            pbar_train.set_description("Train: Avg lsoss: {:.3f}, Avg_dice: {:.3f}".\
                format(avg_loss/count,avg_dscoeff/count))
            pbar_train.update(1)
                
        dscoeffs_train.append(np.mean(np.array(dscoeffs), axis = 0))
        # print(dscoeffs_train)
        avg_losses.append(avg_loss/len(train_loader))
        avg_dscoeffs.append(avg_dscoeff/len(train_loader))
        pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}"\
            .format(avg_loss/len(train_loader),avg_dscoeff/len(train_loader)))
        wandb.log({"Train Avg loss": round(avg_loss/len(train_loader), 3), "Train Avg dice": round(avg_dscoeff/len(train_loader), 3)})
        new = list(dscoeffs_train[-1])
        new.append(avg_dscoeff/len(train_loader))
        table_train.append(new)
        model.scheduler.step(avg_loss/len(train_loader))

        val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
                model.validate_model(model.student, wandb, args.n_classes, validation_loader, table_val)
        pbar_outer.update(1)

    table_train = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table": table_train})
    wandb.log({"Val Table": table_val})


for lamb1 in lambda1:
    for lamb2 in lambda2:
        args.pi = True
        args.pa = True
        model = KD_model(args, lamb1, lamb2)
        wandb.init(project='KD_ERFNet_spine', 
                name='ErfNet_pa_'+str(lamb1)+'_'+str(lamb2)+'_'+str(40),
                reinit=True,
                config = args
        )

        columns=["Background", "Anterior", "Posterior", "Avg Dice"]     

        losses = []
        dscoeffs_train = []
        avg_losses = []
        avg_dscoeffs = []

        table_train = []
        table_val = []

        avg_loss = 0.0
        avg_dscoeff = 0.0
        val_avg_loss = 0.0
        val_avg_dscoeff = 0.0

        pbar_outer = tqdm(total=args.n_epochs, desc = 'Epoch count', leave=False)
        pbar_train = tqdm(total = len(train_loader), desc = "Train: Avg loss{}, avg_dice{}"\
                .format(avg_loss,avg_dscoeff), leave=False)

        for epoch in range(20):
            pbar_train.n = 0
            pbar_train.refresh()

            avg_loss = 0.0
            avg_dscoeff = 0.0
            
            count = 0
            dscoeffs = []

            for step, batch in enumerate(train_loader):
                count+=1
                loss, dscoeff = model.train_loop(batch)
                dscoeffs.append(dscoeff)
                avg_loss += loss.item()
                avg_dscoeff += sum(dscoeff)/(args.n_classes)
                pbar_train.set_description("Train: Avg lsoss: {:.3f}, Avg_dice: {:.3f}".\
                    format(avg_loss/count,avg_dscoeff/count))
                pbar_train.update(1)
                    
            dscoeffs_train.append(np.mean(np.array(dscoeffs), axis = 0))
            # print(dscoeffs_train)
            avg_losses.append(avg_loss/len(train_loader))
            avg_dscoeffs.append(avg_dscoeff/len(train_loader))
            pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}"\
                .format(avg_loss/len(train_loader),avg_dscoeff/len(train_loader)))
            wandb.log({"Train Avg loss": round(avg_loss/len(train_loader), 3), "Train Avg dice": round(avg_dscoeff/len(train_loader), 3)})
            new = list(dscoeffs_train[-1])
            new.append(avg_dscoeff/len(train_loader))
            table_train.append(new)
            model.scheduler.step(avg_loss/len(train_loader))

            val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
                    model.validate_model(model.student, wandb, args.n_classes, validation_loader, table_val)
            pbar_outer.update(1)

        table_train = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
        table_val = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
        wandb.log({"Train Table": table_train})
        wandb.log({"Val Table": table_val})