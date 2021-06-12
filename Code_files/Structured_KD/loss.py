import torch
import torch.nn.functional as F
from tqdm import tqdm


import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def dice_coeff(y_true, y_pred):
    smooth = 1
    # print(y_true.shape, y_pred.shape)
    # assert y_true.shape == y_pred.shape, "Tensor dimensions must match"
    shape = y_true.shape
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    intersection = torch.sum(y_true_flat * y_pred_flat)
    score = (2. * intersection + smooth) / (torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth)
    return score

def dice_coeff_multiclass(y_true, y_pred, num_classes, subnet = False):
    if not subnet:
        dice = []
        output = torch.argmax(y_pred, dim=1).unsqueeze(dim=1)
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
    else:
        dice = []
        for i in range(num_classes):
            segs = y_true.clone().detach()
            segs[y_true==i]=1
            segs[y_true!=i]=0
            # print(torch.unique(segs==y_true))
            outs = y_pred.clone().detach()
            outs[y_pred==i]=1
            outs[y_pred!=i]=0
            # print(torch.unique(outs==output))
            dice.append(dice_coeff(segs, outs).item())
        # print(dice)
        return dice, y_pred, y_true

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

def cross_ent_dice_loss(y_true, y_pred):
    loss = F.cross_entropy(y_pred.float(), y_true) + dice_loss(y_true, y_pred)
    return loss

def Cross_ent_dist(predicted, target):
    return -(target * torch.log(predicted)).sum(dim=1).mean()

def pixel_wise_KL(pred_student, teacher_pred):
    criterion = torch.nn.KLDivLoss()
    loss = criterion(pred_student.log(), teacher_pred)
    return loss

def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def pair_wise_loss(feat_S, feat_T):
    # feat_S = preds_S
    # feat_T = preds_T
    feat_T.detach()

    total_w, total_h = feat_T.shape[2], feat_T.shape[3]
    # patch_w, patch_h = int(total_w*scale), int(total_h*scale)
    # maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
    loss = sim_dis_compute(feat_S, feat_T)
    return loss