from models.unet_3_enc import UNet2D
from models.mo_unet_3_enc import MO_Net_decoder, MO_Net_encoder
from utils import *
from torch.autograd import Variable
import numpy as np
import pandas as pd

def train_UNet(input_type, output_type, wandb, train_loader, validation_loader, weights, opt, device, model_name):
    epochs = opt.n_epochs
    num_classes = opt.n_classes


    model = UNet2D(opt.n_classes, opt.n_channels).cuda()
    # wandb.watch(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)


    if opt.n_classes == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if opt.n_classes == 2:
        columns=["Background", output_type, "Avg Dice"]        
    model, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model, wandb, epochs, num_classes, weights, train_loader, input_type, output_type, columns, validation_loader, \
            optimizer, scheduler, phase = str(1), part = 'f', model_name = opt.model_type, filepath = model_name, save = 1)
    
    table_train = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table": table_train})
    wandb.log({"Val Table": table_val})


def train_MOUNet(input_type, output_type_phase1, output_type_phase2, wandb, train_loader, validation_loader, weights, opt, device, model_name):
    epochs1 = opt.n_epochs_phase1
    epochs2 = opt.n_epochs_phase2
    num_classes_1 = opt.n_classes_phase1
    num_classes_2 = opt.n_classes_phase2

    #phase1 training

    model1 = UNet2D(num_classes_1, opt.n_channels).cuda()
    # wandb.watch(model)
    optimizer = torch.optim.Adam(model1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)


    if num_classes_1== 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_1 == 2:
        columns=["Background", output_type_phase1, "Avg Dice"]

    print('Training Phase 1 left')    
    model1, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model1, wandb, epochs1, num_classes_1, weights[0:2], train_loader, input_type, output_type_phase1, columns, \
            validation_loader, optimizer, scheduler, part = 'l', save = 0)
    
    table_train_1 = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_1 = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase1": table_train_1})
    wandb.log({"Val Table_phase1": table_val_1})

    #phase2 training

    model_enc = MO_Net_encoder(opt.n_channels).cuda()
    model_dec1 = MO_Net_decoder(num_classes_1).cuda()
    model_dec2 = MO_Net_decoder(num_classes_2).cuda()

    params1 = model1.state_dict()
    params2 = model_enc.state_dict()
    params3 = model_dec1.state_dict()
    params4 = model_dec2.state_dict()

    for item in params1:
        if item in params2:
            params2[item] = params1[item]
    for item in params1:
        if item in params3:
            params3[item] = params1[item]
    for item in params1:
        if item in params4:
            if (params4[item].shape == params1[item].shape):
                params4[item] = params1[item]
    model_enc.load_state_dict(params2)
    model_dec1.load_state_dict(params3)
    model_dec2.load_state_dict(params4)
    optimizer0 = torch.optim.Adam(model_enc.parameters(), lr = 0.000003)
    optimizer1 = torch.optim.Adam(model_dec1.parameters(), lr = 0.000001)
    optimizer2 = torch.optim.Adam(model_dec2.parameters(), lr = 0.00001)

    # model2 = UNet2D(opt.num_classes_2, opt.n_channels).cuda()
    # wandb.watch(model)
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler0 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer0, factor=0.5, patience=2, min_lr=1e-6)
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, factor=0.5, patience=2, min_lr=1e-6)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, factor=0.5, patience=2, min_lr=1e-6)


    if num_classes_2 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_2 == 2:
        columns=["Background", output_type_phase2, "Avg Dice"]

    print('Training Phase 2 full')   
    model_enc, model_dec1, model_dec2, avg_losses, avg_dscoeffs, dscoeffs_train, \
        table_train_2, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val_2 = \
        train_val_mounet(model_enc, model_dec1, model_dec2, wandb, epochs2, num_classes_1, num_classes_2,\
     weights, train_loader, validation_loader, input_type, output_type_phase1, output_type_phase2, columns, \
         optimizer0, optimizer1, optimizer2, scheduler0, scheduler1, scheduler2, 0.5, model_name = opt.model_type, filepath = model_name)

    table_train_2 = wandb.Table(dataframe=pd.DataFrame(table_train_2, columns = columns))
    table_val_2 = wandb.Table(dataframe=pd.DataFrame(table_val_2, columns = columns))
    wandb.log({"Train Table_phase2": table_train_2})
    wandb.log({"Val Table_phase2": table_val_2})

def train_NFTNet(input_type, output_type_phase1_left, output_type_phase1_right, output_type_phase2, wandb, train_loader, validation_loader, weights, opt, device, model_name):
    epochs1 = opt.n_epochs_phase1
    epochs2 = opt.n_epochs_phase2
    num_classes_1 = opt.n_classes_phase1
    num_classes_2 = opt.n_classes_phase2

    #phase1 training NFTs
    
    model1_left = UNet2D(num_classes_1, opt.n_channels).cuda()
    model1_right = UNet2D(num_classes_1, opt.n_channels).cuda()
    # wandb.watch(model)
    optimizer_left = torch.optim.Adam(model1_left.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler_left = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_left, factor=0.5, patience=2, min_lr=1e-6)
    optimizer_right = torch.optim.Adam(model1_right.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler_right = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_right, factor=0.5, patience=2, min_lr=1e-6)


    if num_classes_1 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_1 == 2:
        columns=["Background", output_type_phase1_left, "Avg Dice"]
    
    print('Training Phase 1 left')       
    model1_left, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model1_left, wandb, epochs1, num_classes_1, weights[0:2], train_loader, input_type, output_type_phase1_left, columns, \
            validation_loader, optimizer_left, scheduler_left, phase = str(1), part = 'l', save = 0)
    
    table_train_1_left = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_1_left = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase1_left": table_train_1_left})
    wandb.log({"Val Table_phase1_left": table_val_1_left})

    if num_classes_1 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_1 == 2:
        columns=["Background", output_type_phase1_right, "Avg Dice"]
    
    print('Training Phase 1 right')
    model1_right, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model1_right, wandb, epochs1, num_classes_1, weights[0:2], train_loader, input_type, output_type_phase1_right, columns, \
            validation_loader, optimizer_right, scheduler_right, part = 'r', save = 0)
    
    table_train_1_right = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_1_right = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase1_right": table_train_1_right})
    wandb.log({"Val Table_phase1_right": table_val_1_right})


    #phase2 training NFT

    model2 = UNet2D(num_classes_2, opt.n_channels).cuda()
    model2 = set_grads_NFT(model1_left, model1_right, model2)

    # wandb.watch(model)
    optimizer = torch.optim.Adam(model2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)


    if num_classes_2 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_2 == 2:
        columns=["Background", output_type_phase2, "Avg Dice"]
    
    print('Training Phase 2 full')
    model2, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model2, wandb, epochs2, num_classes_2, weights, train_loader, input_type, output_type_phase2, columns, validation_loader, \
            optimizer, scheduler, phase = str(2), part = 'f', model_name = opt.model_type, filepath = model_name, save = 1)
    
    table_train_2 = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_2 = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase2": table_train_2})
    wandb.log({"Val Table_phase2": table_val_2})

def train_SubNet(input_type, output_type_phase1,  output_type_phase2_full, wandb, train_loader, validation_loader, weights, opt, device, model_name):
    epochs1 = opt.n_epochs_phase1
    epochs2 = opt.n_epochs_phase2
    num_classes_1 = opt.n_classes_phase1
    # num_classes_2 = opt.n_classes_phase2

    #phase1 training NFTs

    model1 = UNet2D(num_classes_1, opt.n_channels).cuda()
    # wandb.watch(model)
    optimizer_left = torch.optim.Adam(model1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler_left = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_left, factor=0.5, patience=2, min_lr=1e-6)

    if num_classes_1 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_1 == 2:
        columns=["Background", output_type_phase1, "Avg Dice"]        
    model1, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
        train_model(model1, wandb, epochs1, num_classes_1, weights[0:2], train_loader, input_type, output_type_phase1, columns, \
            validation_loader, optimizer_left, scheduler_left, phase = str(1), part = 'l', model_name = opt.model_type, filepath = model_name + '_left_', save = 1)
    
    table_train_1_left = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_1_left = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase1_left": table_train_1_left})
    wandb.log({"Val Table_phase1_left": table_val_1_left})


    #phase2 training NFT

    model2 = UNet2D(num_classes_1, opt.n_channels).cuda()

    optimizer = torch.optim.Adam(model2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, min_lr=1e-6)


    if num_classes_1 == 3:
        columns=["Background", "Anterior", "Posterior", "Avg Dice"]
    if num_classes_1 == 2:
        columns=["Background", output_type_phase2_full, "Avg Dice"]        
    model2, avg_losses, avg_dscoeffs, dscoeffs_train, table_train_2, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val_2 = \
        train_model(model2, wandb, epochs2, num_classes_1, weights[0:2], train_loader, input_type, output_type_phase2_full, columns, \
            validation_loader, optimizer, scheduler, phase = str(1), part = 'l', model_name = opt.model_type, filepath = model_name + '_full_', save = 1)
    
    table_train_2 = wandb.Table(dataframe=pd.DataFrame(table_train, columns = columns))
    table_val_2 = wandb.Table(dataframe=pd.DataFrame(table_val, columns = columns))
    wandb.log({"Train Table_phase2": table_train_2})
    wandb.log({"Val Table_phase2": table_val_2})

    Avg_right_dice = eval_subnet_right(model2, model1, num_classes_1, validation_loader, input_type, output_type_phase1, 'seg_right', output_type_phase2_full, wandb)
    print(Avg_right_dice)