"""
Helper functions for training and testing, for the moment many things contain absolute paths and is quite dirty
The thing is, I'll be improving the code as the research advances
"""

import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
# from torch.utils.checkpoint import *
# for gradient checkpointing (doesn't seems so straightforward) check:
# https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
# https://github.com/pytorch/pytorch/pull/4594
from torch.utils.tensorboard import SummaryWriter

from apex import amp, optimizers


# Gradient checkpointing .... TODO when/if necessary


# from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(data, n, dim=0):
    """Yield successive n-sized chunks from data by the dimension dim"""
    for i in range(0, data.shape[dim], n):
        yield data[i:i + n, :, :]


def loss_txt2txt_multi(prediction, target,
                       pred_orig_lang, tgt_orig_lang,
                       pred_dest_lang, tgt_dest_lang,
                       pred_task_lang, tgt_task_lang,
                       pred_task_desc, tgt_task_desc,
                       scale_loss=1.,
                       scale_orig_lang_loss=1.,
                       scale_dest_lang_loss=1.,
                       scale_task_lang_loss=1.,
                       scale_task_desc_loss=1.,
                       fn_loss=F.nll_loss,  # fn_loss=F.kl_div
                       ):
    """
    Computes the complete loss for all the target task and meta tasks
    predictions and targets MUST be LONG format (like pre-embedding) for nll_loss and kl_div
    :param prediction:
    :param target:
    :param pred_orig_lang:
    :param tgt_orig_lang:
    :param pred_dest_lang:
    :param tgt_dest_lang:
    :param pred_task_lang:
    :param tgt_task_lang:
    :param pred_task_desc:
    :param tgt_task_desc:
    :param scale_loss:
    :param scale_orig_lang_loss:
    :param scale_dest_lang_loss:
    :param scale_task_lang_loss:
    :param scale_task_desc_loss:
    :param fn_loss: 
    :return:
    """
    pred_loss = fn_loss(prediction, target) * scale_loss
    orig_lang_loss = fn_loss(pred_orig_lang, tgt_orig_lang) * scale_orig_lang_loss
    dest_lang_loss = fn_loss(pred_dest_lang, tgt_dest_lang) * scale_dest_lang_loss
    task_lang_loss = fn_loss(pred_task_lang, tgt_task_lang) * scale_task_lang_loss
    task_desc_loss = fn_loss(pred_task_desc, tgt_task_desc) * scale_task_desc_loss
    loss = pred_loss + orig_lang_loss + dest_lang_loss + task_lang_loss + task_desc_loss
    return loss, (pred_loss, orig_lang_loss, dest_lang_loss, task_lang_loss, task_desc_loss)


def main(model, optimizer='FusedAdam', lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
         amsgrad=False, adam_w_mode=True, max_grad_norm=1.0):

    if optimizer == 'FusedLAMB':  # designed for BERT to augment the batch sizes and decrease training time
        optimizer = optimizers.FusedLAMB(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                         amsgrad=False, adam_w_mode=adam_w_mode, max_grad_norm=max_grad_norm)
    elif optimizer == 'FusedNovoGrad':  # takes less memory than Adam
        optimizer = optimizers.FusedNovoGrad(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                             amsgrad=amsgrad)
    else:  # default is FusedADAM
        optimizer = optimizers.FusedAdam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                         amsgrad=False,  # NOT SUPPORTED in FusedAdam!
                                         adam_w_mode=adam_w_mode,
                                         )
    data_loader = ...  # TODO this one is the tough one
    criterion = loss_txt2txt_multi
    train_main(model, optimizer, data_loader, criterion)


def train_main(model, optimizer, data_loader, criterion=loss_txt2txt_multi,
               opt_level="O1", keep_batchnorm_fp32=True, loss_scale="dynamic"  # recommended params: O1, True, dynamic
               ):
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level,  # [O0, O1, O2, O3 ]
                                      keep_batchnorm_fp32=keep_batchnorm_fp32,
                                      loss_scale=loss_scale)
    writer = SummaryWriter()
    batch_count = 0
    epoch_count = 0
    # get epoch iterator from data loader
    for epoch_iter in data_loader.load():
        # get data iterator for the epoch from the data loader
        for train_data_iter, test_data_iter in epoch_iter:
            batch_count = train_epoch(model, optimizer, criterion, train_data_iter, epoch_count, writer, batch_count)
            test_epoch(model, criterion, test_data_iter, epoch_count, writer, batch_count)
        epoch_count += 1
    pass


def train_epoch(model, optimizer, criterion, data_iterator, epoch, summary_writer, total_batch_count=0):
    """

    :param model:
    :param optimizer:
    :param criterion:
    :param data_iterator:
    :param epoch:
    :param summary_writer:
    :param total_batch_count:
    :return:
    """
    torch.cuda.empty_cache()
    model.train()
    train_loss = 0
    batch_idx = 1
    names = ['pred_loss', 'orig_lang_loss', 'dest_lang_loss', 'task_lang_loss', 'task_desc_loss']
    indiv_losses_acc = np.array([0.] * 5)
    for metadata, data, label in data_iterator:
        orig_lang, dest_lang, task_lang, task_desc = metadata

        optimizer.zero_grad()
        out = model(data)
        pred, pred_olang, pred_dlang, pred_tlang, pred_tdesc = out
        loss, ind_losses = criterion(pred, label,
                                     pred_olang, orig_lang,
                                     pred_dlang, dest_lang,
                                     pred_tlang, task_lang,
                                     pred_tdesc, task_desc,
                                     )
        # pred_loss, orig_lang_loss, dest_lang_loss, task_lang_loss, task_desc_loss = ind_losses
        # Write summaries and details on TensorBoard for the task
        summary_writer.add_scalar("Loss/train", loss.data.item(), global_step=total_batch_count + batch_idx)
        acc = []
        for n, l in zip(names, ind_losses):
            ldata = l.data.item()
            acc.append(ldata)
            summary_writer.add_scalar("Loss/train-{}".format(n), ldata)
        acc = np.array(acc)  # keep track of each individual part of the loss to log later

        indiv_losses_acc = indiv_losses_acc + acc
        with amp.scaled_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        batch_idx += 1
        torch.cuda.empty_cache()

    summary_writer.add_scalar("EpochLoss/train", train_loss / batch_idx, epoch)
    for n, l in zip(names, indiv_losses_acc):
        summary_writer.add_scalar("EpochLoss/train-{}".format(n), l)
    print('====> Timestamp {} Epoch: {} Average loss: {:.8f}'.format(datetime.now(), epoch, train_loss / batch_idx))
    return batch_idx + total_batch_count


def test_epoch(model, criterion, data_iterator, epoch, summary_writer, total_batch_count=0):
    """

    :param model:
    :param criterion:
    :param data_iterator:
    :param epoch:
    :param summary_writer:
    :param total_batch_count:
    :return:
    """
    torch.cuda.empty_cache()
    model.eval()
    test_loss = 0
    batch_idx = 1
    names = ['pred_loss', 'orig_lang_loss', 'dest_lang_loss', 'task_lang_loss', 'task_desc_loss']
    indiv_losses_acc = np.array([0.] * 5)
    for metadata, data, label in data_iterator:
        orig_lang, dest_lang, task_lang, task_desc = metadata
        out = model(data)
        pred, pred_olang, pred_dlang, pred_tlang, pred_tdesc = out
        loss, ind_losses = criterion(pred, label,
                                     pred_olang, orig_lang,
                                     pred_dlang, dest_lang,
                                     pred_tlang, task_lang,
                                     pred_tdesc, task_desc,
                                     )
        # pred_loss, orig_lang_loss, dest_lang_loss, task_lang_loss, task_desc_loss = ind_losses
        # Write summaries and details on TensorBoard for the task
        summary_writer.add_scalar("Loss/test", loss.data.item(), global_step=total_batch_count + batch_idx)
        acc = []
        for n, l in zip(names, ind_losses):
            ldata = l.data.item()
            acc.append(ldata)
            summary_writer.add_scalar("Loss/test-{}".format(n), ldata)
        acc = np.array(acc)  # keep track of each individual part of the loss to log later

        indiv_losses_acc = indiv_losses_acc + acc
        batch_idx += 1
        torch.cuda.empty_cache()

    summary_writer.add_scalar("EpochLoss/test", test_loss / batch_idx, epoch)
    for n, l in zip(names, indiv_losses_acc):
        summary_writer.add_scalar("EpochLoss/test-{}".format(n), l)
    print('====> Timestamp {} Epoch: {} Test Set loss: {:.8f}'.format(datetime.now(), epoch, test_loss / batch_idx))
    return batch_idx + total_batch_count
