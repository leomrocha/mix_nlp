"""
Helper functions for training and testing, for the moment many things contain absolute paths and is quite dirty
The thing is, I'll be improving the code as the research advances
"""

import os
import sys
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from apex import amp, optimizers
# from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex.multi_tensor_apply import multi_tensor_applier
amp.initialize()
# APEX examples
# import torch
# import amp
# model = ...
# optimizer = ...
# model, optimizer = amp.initialize(model, optimizer,
#                                       opt_level=args.opt_level,
#                                       keep_batchnorm_fp32=args.keep_batchnorm_fp32,
#                                       loss_scale=args.loss_scale
#                                       )
# model, optimizer = amp.initialize(model, optimizer, opt_level="O1")  # [O0, O1, O2, O3 ]
# for data, label in data_iter:
#     out = model(data)
#     loss = criterion(out, label)
#     optimizer.zero_grad()
#     with amp.scaled_loss(loss, optimizer) as scaled_loss:
#         scaled_loss.backward()
# optimizer.step()

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
    :param pred_task:
    :param tgt_task:
    :param scale_loss:
    :param scale_orig_lang_loss:
    :param scale_dest_lang_loss:
    :param scale_task_lang_loss:
    :param scale_task_loss:
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


def loss_txt2txt_single(prediction, target,
                        fn_loss=F.nll_loss,  # fn_loss=F.kl_div
                        ):
    """
    Simple Text-to-Text loss between an input and an output
    
    :param prediction:
    :param target: the target to measure a long, having indices (as pre-embedding ones)
    :param fn_loss: loss function, use nll_loss or kl_div for this to work correctly
    :return:
    """
    loss = fn_loss(prediction, target)
    return loss


# writer = SummaryWriter()


# model = ...
# optimizer = ...
# model, optimizer = amp.initialize(model, optimizer,
#                                       opt_level=args.opt_level,
#                                       keep_batchnorm_fp32=True,
#                                       loss_scale="dynamic"
#                                       )
# model, optimizer = amp.initialize(model, optimizer, opt_level="O2")  # [O0, O1, O2, O3 ]
# for data, label in data_iter:
#     out = model(data)
#     loss = criterion(out, label)
#     optimizer.zero_grad()
#     with amp.scaled_loss(loss, optimizer) as scaled_loss:
#         scaled_loss.backward()
#     optimizer.step()

def train_main(model, optimizer, criterion, data_loader,
               opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic"
               ):

    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level,  # [O0, O1, O2, O3 ]
                                      keep_batchnorm_fp32=keep_batchnorm_fp32,
                                      loss_scale=loss_scale)


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
    train_loss = 0
    batch_idx = 1
    names = ['pred_loss', 'orig_lang_loss', 'dest_lang_loss', 'task_lang_loss', 'task_desc_loss']
    indiv_losses_acc = np.array([0.]*5)
    for metadata, data, label in data_iterator:
        orig_lang, dest_lang, task_lang, task = metadata
        out = model(data)
        loss, ind_losses = criterion(out, label)
        # pred_loss, orig_lang_loss, dest_lang_loss, task_lang_loss, task_desc_loss = ind_losses
        # Write summaries and details on TensorBoard for the task
        summary_writer.add_scalar("Loss/train", loss.data.item(), global_step= total_batch_count + batch_idx)
        acc = []
        for n, l in zip(names, ind_losses):
            ldata = l.data.item()
            acc.append(ldata)
            summary_writer.add_scalar("Loss/train-{}".format(n), ldata)
        acc = np.array(acc)  # keep track of each individual part of the loss to log later

        indiv_losses_acc = indiv_losses_acc + acc
        optimizer.zero_grad()
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


