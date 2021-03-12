import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.unet_3_enc import UNet2D
from models.mo_unet_3_enc import MO_Net_decoder, MO_Net_encoder

import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd

def dice_coeff(y_true, y_pred):
    smooth = 1
    # print(y_true.shape, y_pred.shape)
    assert y_true.shape == y_pred.shape, "Tensor dimensions must match"
    shape = y_true.shape
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = torch.sum(y_true_flat * y_pred_flat)
    score = (2. * intersection + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)
    return score

def dice_coeff_multiclass(y_true, y_pred, num_classes):
    dice = []
    output = torch.argmax(y_pred, dim=1).unsqueeze(dim=1)
    # print(output.shape)
    # y_true = y_true.squeeze(0)
    # print(torch.unique(y_true), torch.unique(output))
    # print(output.shape, y_true.shape)
    for i in range(num_classes):
        segs = y_true.clone().detach()
        segs[y_true==i]=1
        segs[y_true!=i]=0
        # print(torch.unique(segs==y_true))
        outs = output.clone().detach()
        outs[output==i]=1
        outs[output!=i]=0
        # print(torch.unique(outs==output))
        dice.append(dice_coeff(segs, outs).item())
    # print(dice)
    return dice, output, y_true

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model..unsqueeze(0)
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

def bce_dice_loss(y_true, y_pred):
    loss = F.binary_cross_entropy(y_pred, y_true) + dice_loss(y_true, y_pred)
    return loss

def cross_ent_dice_loss(y_true, y_pred, weights):
    # print(y_true.shape, y_pred.shape)
    # loss = nn.CrossEntropyLoss
    # y_pred = torch.argmax(y_pred, 1)
    loss = F.cross_entropy(y_pred.float(), y_true, weight=weights) + dice_loss(y_true, y_pred)
    return loss


def plot_output(input_img, segs, outs, path):
    fig = plt.figure()
    plt.subplot(3,1,1).imshow(input_img[0][0].squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(3,1,1).set_title('Scan')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,2).imshow(segs.squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(3,1,2).set_title('GT')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3,1,3).imshow(outs.squeeze().cpu().numpy(), cmap='gray')
    plt.subplot(3,1,3).set_title('Seg_op')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # plt.show()
    plt.savefig(path)

def load_models(n_channels, n_classes, model_type, unet_path = 'none', model_1_path = 'none', model_2_path = 'none'):
    if model_type == 'unet':
        model = UNet2D(n_classes, n_channels).cuda()
        model.load_state_dict(torch.load(unet_path, map_location=cuda()))
        return model
    if model_type == 'mounet':
        model_enc = MO_Net_encoder(n_channels).cuda()
        model_dec = MO_Net_decoder(n_classes).cuda()
        model_enc.load_state_dict(torch.load(model_1_path, map_location=cuda()))
        model_dec.load_state_dict(torch.load(model_2_path, map_location=cuda()))
        return model_enc, model_dec
    if model_type == 'nftnet':
        model = UNet2D(n_classes, n_channels).cuda()
        model.load_state_dict(torch.load(unet_path, map_location=device))
        return model
    if model_type == 'subnet':
        model_full = UNet2D(n_classes, n_channels).cuda()
        model_left = UNet2D(n_classes, n_channels).cuda()
        model_full.load_state_dict(torch.load(model_1_path, map_location=cuda()))
        model_left.load_state_dict(torch.load(model_2_path, map_location=cuda()))
        return model_full, model_left

def set_grads_NFT(model1, model2, model3):
    params1 = model1.state_dict()
    filter_list_1 = []
    for item in params1:
        if params1[item].type() == 'torch.cuda.FloatTensor':
            params1[item].requires_grad = True
        if (params1[item].requires_grad == True):
            filter_list_1.append(params1[item])
    
    params2 = model2.state_dict()
    filter_list_2 = []
    for item in params2:
        if params2[item].type() == 'torch.cuda.FloatTensor':
            params2[item].requires_grad = True
        if (params2[item].requires_grad == True):
            filter_list_2.append(params2[item])

    diff_norm = []
    for i in range(len(filter_list_1)):
        new_ten = filter_list_1[i] - filter_list_2[i]
        diff_norm.append(torch.norm(new_ten))
    
    count = 0
    for item in params1:
        if params1[item].type() == 'torch.cuda.FloatTensor':
            if diff_norm[count].cpu() > 15:
                params1[item].requires_grad = True
            else:
                params1[item].requires_grad = False
            count+=1
    
    params3 = model3.state_dict()

    for item in params1:
        if item in params3:
            if (params3[item].shape == params1[item].shape):
                params3[item] = params1[item]
                params3[item].requires_grad = params1[item].requires_grad

    return model3

def train_model(model, wandb, epochs, num_classes, weights, train_loader, input_type, output_type, columns, validation_loader, \
    optimizer, scheduler, phase = '1', part = 'l', model_name='none', filepath='none'):

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

    pbar_outer = tqdm(total=epochs, desc = 'Epoch count', leave=False)
    pbar_train = tqdm(total = len(train_loader), desc = "Train: Avg loss{}, avg_dice{}"\
            .format(avg_loss,avg_dscoeff), leave=False)
    
    for epoch in range(1, epochs+1):
        pbar_train.n = 0
        pbar_train.refresh()
        model.train()
        # for m in model.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.track_running_stats=True

        avg_loss = 0.0
        avg_dscoeff = 0.0
        
        count = 0
        dscoeffs = []
        for i, batch in enumerate(train_loader):
            count+=1
            optimizer.zero_grad()
            input_img = Variable(batch[input_type]).cuda()
            segs = Variable(batch[output_type]).cuda()
            outputs = model(input_img)
            
            loss = cross_ent_dice_loss(torch.squeeze(segs, dim=1), outputs, weights)
            loss.backward()
            optimizer.step()
            dscoeff, outs, segs = dice_coeff_multiclass(segs, outputs, num_classes)
            dscoeffs.append(dscoeff)
            

            avg_loss += loss.item()
            avg_dscoeff += sum(dscoeff)/(num_classes)
            pbar_train.set_description("Train: Avg lsoss: {:.3f}, Avg_dice: {:.3f}".\
                format(avg_loss/count,avg_dscoeff/count))
            pbar_train.update(1)
            
        dscoeffs_train.append(np.mean(np.array(dscoeffs), axis = 0))
        avg_losses.append(avg_loss/len(train_loader))
        avg_dscoeffs.append(avg_dscoeff/len(train_loader))
        pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}"\
            .format(avg_loss/len(train_loader),avg_dscoeff/len(train_loader)))
        wandb.log({"Train Avg loss"+'_'+phase+'_'+part: round(avg_loss/len(train_loader), 3), "Train Avg dice"+'_'+phase+'_'+part: round(avg_dscoeff/len(train_loader), 3)})
        new = list(dscoeffs_train[-1])
        new.append(avg_dscoeff/len(train_loader))
        table_train.append(new)
        scheduler.step(avg_loss/len(train_loader))


        
        val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val = \
            validate_model(model, wandb, num_classes, weights, validation_loader, input_type, output_type, table_val, phase, part)
        pbar_outer.update(1)
        filepath = '../models/' + model_name + '/' + filepath + 'epoch_' + epoch + '.pt'
        torch.save(model.state_dict(), filepath)
    
    return model, avg_losses, avg_dscoeffs, dscoeffs_train, table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val

def validate_model(model, wandb, num_classes, weights, validation_loader, input_type, output_type, table_val, phase = '1', part = 'l'):

    val_losses = []
    val_dscoeffs = []
    val_avg_losses = []
    val_avg_dscoeffs = []

    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0

    
    pbar_val = tqdm(total = len(validation_loader), desc = "Val: Avg loss{}, avg_dice{}"\
            .format(val_avg_loss,val_avg_dscoeff), leave=False)
    pbar_val.n = 0
    # pbar_val.refresh()
    # model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.track_running_stats=False
    
    count = 0
    val_dscoeffs = []
    for i, batch in enumerate(validation_loader):
        with torch.no_grad():
            count+=1
            input_img = Variable(batch[input_type]).cuda()
            segs = Variable(batch[output_type]).cuda()
            outputs = model(input_img)
            
            loss = cross_ent_dice_loss(torch.squeeze(segs, dim=1), outputs, weights)
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
    wandb.log({"Val Avg loss"+'_'+phase+'_'+part: round(val_avg_loss/len(validation_loader), 3), "Val Avg dice"+'_'+phase+'_'+part: round(val_avg_dscoeff/len(validation_loader), 3)})
    new = list(val_dscoeffs[-1])
    new.append(val_avg_dscoeff/len(validation_loader))
    table_val.append(new)

    return val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val

def train_val_mounet(model_enc, model_dec1, model_dec2, wandb, epochs, num_classes_1, num_classes_2,\
     weights, train_loader, validation_loader, input_type, output_type1, output_type2, columns, optimizer0, \
         optimizer1, optimizer2, scheduler0, scheduler1, scheduler2, Lambda=0.5, model_name='none', filepath='none'):

    losses = []
    dscoeffs_train = []
    avg_losses = []
    avg_dscoeffs = []

    table_train = []
    table_val = []
    
    avg_loss = 0.0
    avg_dscoeff = 0.0

    val_losses = []
    val_dscoeffs = []
    val_avg_losses = []
    val_avg_dscoeffs = []

    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0

    pbar_epoch = tqdm(total=epochs, desc = 'Epoch count', leave=False)
    pbar_train = tqdm(total = len(train_loader), desc = "Train: Avg loss{}, avg_dice{}".format(avg_loss,avg_dscoeff), leave=False)
    pbar_val = tqdm(total = len(validation_loader), desc = "Val: Avg loss{}, avg_dice{}".format(avg_loss,avg_dscoeff), leave=False)
    for epoch in range(1, epochs+1):
        # print('Epoch: ', epoch)
        model_enc.train()
        model_dec1.train()
        model_dec2.train()
        avg_loss = 0.0
        avg_dscoeff = 0.0

        count = 0
        dscoeffs = []
        pbar_train.n = 0
        for i, batch in enumerate(train_loader):
            count+=1
            optimizer0.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
    #         print(len(batch[0]))model_name
            optimizer0.step()
            optimizer1.step()
            optimizer2.step()

            dscoeff, outs, segs = dice_coeff_multiclass(segs2, outputs2, num_classes_2)
            dscoeffs.append(dscoeff)
            avg_loss += loss.item()
            avg_dscoeff += sum(dscoeff)/(num_classes_2)
            pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}".format(avg_loss/count,avg_dscoeff/count))
            pbar_train.update(1)
        dscoeffs_train.append(np.sum(np.array(dscoeffs), axis=0)/len(train_loader))
        avg_losses.append(avg_loss/len(train_loader))
        avg_dscoeffs.append(avg_dscoeff/len(train_loader))
        # print(len(dscoeffs_train[-1]))
        new = list(dscoeffs_train[-1])
        new.append(avg_dscoeff/len(train_loader))
        table_train.append(new)
        scheduler0.step(avg_loss/len(train_loader))
        scheduler1.step(avg_loss/len(train_loader))
        scheduler2.step(avg_loss/len(train_loader))
        pbar_train.set_description("train: Avg loss: {:.3f}, Avg_dice: {:.3f}".format(avg_loss/len(train_loader),avg_dscoeff/len(train_loader)))
        wandb.log({"Train Avg loss phase2": round(avg_loss/len(train_loader), 3), "Train Avg dice phase2": round(avg_dscoeff/len(train_loader), 3)})

        filepath_enc = '../models/' + model_name + '/' + 'enc_' + filepath + 'epoch_' + epoch + '.pt'
        torch.save(model_enc.state_dict(), filepath_enc)
        filepath_dec = '../models/' + model_name + '/' + 'dec_' + filepath + 'epoch_' + epoch + '.pt'
        torch.save(model_dec2.state_dict(), filepath_dec)
    #         pbar.set_postfix(**{'loss (batch)': loss.item(), 'DSC (batch)': dscoeff})
    #         pbar.update(i)

        # # model0.eval()
        val_avg_loss = 0.0
        val_avg_dscoeff = 0.0
        
        count = 0
        val_dscoeffs = []
        pbar_val.n = 0
        for i, batch in enumerate(validation_loader):
            with torch.no_grad():
                count+=1
                # optimizer.zero_grad()
        #         print(len(batch[0]))
                input_img = Variable(batch[input_type]).cuda()
                segs = Variable(batch[output_type1]).cuda()
                segs2 = Variable(batch[output_type2]).cuda()
                encoder0, encoder1, encoder2, encoder3, center = model_enc(input_img)
                outputs1 = model_dec1(encoder0, encoder1, encoder2, encoder3, center)
                outputs2 = model_dec2(encoder0, encoder1, encoder2, encoder3, center)
                # print(outputs.shape)
                # print(torch.unique(torch.argmax(outputs, 1)))
                
                loss1 = cross_ent_dice_loss(torch.squeeze(segs, dim=1), outputs1, weights[0:2])
                loss2 = cross_ent_dice_loss(torch.squeeze(segs2, dim=1), outputs2, weights)
                loss = ((1 - Lambda) * loss1) + (Lambda * loss2)
                # loss.backward()   
                # loss.backward()
                # optimizer.step()
                dscoeff, outs, segs = dice_coeff_multiclass(segs2, outputs2, num_classes_2)
            
                # val_losses.append(loss.item())
                val_dscoeffs.append(dscoeff)
                val_avg_loss += loss.item()
                val_avg_dscoeff += sum(dscoeff)/(num_classes_2)
                pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".format(val_avg_loss/count,val_avg_dscoeff/count))
                pbar_val.update(1)
        val_dscoeffs.append(np.sum(np.array(val_dscoeffs), axis=0)/len(validation_loader))
        val_avg_losses.append(val_avg_loss/len(validation_loader))
        val_avg_dscoeffs.append(val_avg_dscoeff/len(validation_loader))
        pbar_val.set_description("Val: Avg loss: {:.3f}, Avg_dice: {:.3f}".\
        format(val_avg_loss/len(validation_loader),val_avg_dscoeff/len(validation_loader)))
        wandb.log({"Val Avg loss phase2": round(val_avg_loss/len(validation_loader), 3), "Val Avg dice phase2": round(val_avg_dscoeff/len(validation_loader), 3)})
        new = list(val_dscoeffs[-1])
        new.append(val_avg_dscoeff/len(validation_loader))
        table_val.append(new)
        pbar_epoch.update(1)

    return model_enc, model_dec1, model_dec2, avg_losses, avg_dscoeffs, dscoeffs_train, \
        table_train, val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val

def eval_subnet_right(model_full, model_left, num_classes, validation_loader, input_type, output_type_left, output_type_right, output_type_full):

    val_losses = []
    val_dscoeffs = []
    val_avg_losses = []
    val_avg_dscoeffs = []

    val_avg_loss = 0.0
    val_avg_dscoeff = 0.0

    
    pbar_val = tqdm(total = len(validation_loader), desc = "Val: Avg loss{}, avg_dice{}"\
            .format(val_avg_loss,val_avg_dscoeff), leave=False)
    pbar_val.n = 0
    # pbar_val.refresh()
    # model.eval()
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.track_running_stats=False
    
    count = 0
    val_dscoeffs = []
    for i, batch in enumerate(validation_loader):
        with torch.no_grad():
            count+=1
            input_img = Variable(batch[input_type]).cuda()
            segs_left = Variable(batch[output_type_left]).cuda()
            segs_right = Variable(batch[output_type_right]).cuda()
            segs_full = Variable(batch[output_type_full]).cuda()

            output_full = model_full(input_img)
            output_left = model_left(input_img)
            output_right = output_full - output_left
            
            # loss = cross_ent_dice_loss(torch.squeeze(segs_full, dim=1), output_full, weights)
            # loss.backward()
            # optimizer.step()
            dscoeff_full, outs_full, segs_full = dice_coeff_multiclass(segs_full, outputs_full, num_classes)
            dscoeff_left, outs_left, segs_left = dice_coeff_multiclass(segs_left, outputs_left, num_classes)
            dscoeff_right, outs_right, segs_right = dice_coeff_multiclass(segs_right, outputs_right, num_classes)
            
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
    wandb.log({"Val Avg loss"+phase+'_'+part: round(val_avg_loss/len(validation_loader), 3), "Val Avg dice"+phase+'_'+part: round(val_avg_dscoeff/len(validation_loader), 3)})
    new = list(val_dscoeffs[-1])
    new.append(val_avg_dscoeff/len(validation_loader))
    table_val.append(new)

    return val_avg_losses, val_avg_dscoeffs, val_dscoeffs, table_val

def test_models(n_channels, n_classes, model_type, device, unet_path = 'none', model_1_path = 'none', model_2_path = 'none'):
    if model_type == 'unet':
        model = load_models(n_channels, n_classes, model_type = model_type, unet_path = unet_path)

    if model_type == 'mounet':
        model_enc, model_dec = load_models(n_channels, n_classes, model_type = model_type, model_1_path = model_1_path, model_2_path = model_2_path)
    if model_type == 'nftnet':
        model = load_models(n_channels, n_classes, model_type = model_type, unet_path = unet_path)
    if model_type == 'subnet':
        model_full, model_left = load_models(n_channels, n_classes, model_type = model_type, model_1_path = model_1_path, model_2_path = model_2_path)
