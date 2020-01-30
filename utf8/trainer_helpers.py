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
#     optimizer.step()


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


def train(model, optimizer, loss_function, batches, epoch, ndatapoints, device, max_seq_len=-1):
    torch.cuda.empty_cache()
    model.train()
    # TODO set the non trainable parameters if needed!!
    train_loss = 0
    #     batch_loss = []
    batch_idx = 1
    for b_data in batches:
        b_train = torch.from_numpy(b_data[:, 0, :].astype("int32")).squeeze().to(device)
        # max_seq_len means I'll take care only of a part of the input to compare output. This is to save computation
        # only and not because of any other reason
        if max_seq_len > 0:
            b_upos = torch.from_numpy(b_data[:, 1, :max_seq_len].astype("int32")).squeeze().to(device)
            b_deprel = torch.from_numpy(b_data[:, 2, :max_seq_len].astype("int32")).squeeze().to(device)
        else:
            b_upos = torch.from_numpy(b_data[:, 1, :].astype("int32")).squeeze().to(device)
            b_deprel = torch.from_numpy(b_data[:, 2, :].astype("int32")).squeeze().to(device)
        #
        optimizer.zero_grad()
        dec = model(b_train)
        # last_latent = latent[-1]
        upos, deprel = dec
        # print(emb.shape,emb.dtype, res.shape, res.dtype)
        # print(upos.shape, b_upos.shape)
        # loss = loss_function(upos, deprel, upos_emb(b_upos), deprel_emb(b_deprel))
        # print("train tensor shapes: ", b_train.shape, upos.shape, b_upos.shape, deprel.shape, b_deprel.shape)
        loss, upos_loss, deprel_loss = loss_function(upos.view([-1, 18]), deprel.view([-1, 278]), b_upos.view([-1]), b_deprel.view([-1]))

        loss.backward()
        train_loss += loss.data.item()  # [0]
        writer.add_scalar("Loss/train", loss.data.item(), global_step=(epoch * batch_idx))
        writer.add_scalar("Loss/train/upos", upos_loss.data.item(), global_step=(epoch * batch_idx))
        writer.add_scalar("Loss/train/deprel", deprel_loss.data.item(), global_step=(epoch * batch_idx))
        optimizer.step()
        batch_idx += 1
        del b_train
        del b_upos
        del b_deprel
        torch.cuda.empty_cache()
    writer.add_scalar("EpochLoss/train", train_loss / batch_idx, epoch)
    print('====> Timestamp {} Epoch: {} Average loss: {:.8f}'.format(datetime.now(), epoch, train_loss / ndatapoints))
    return train_loss


def test(model, loss_function, test_data, epoch, device, max_data=40, max_seq_len=-1):
    """

    :param model:
    :param loss_function:
    :param test_data:
    :param epoch:
    :param device:
    :param max_data: maximum amout of data to test (default 50 due to gpu memory constraints in my pc)
    :return:
    """
    model.eval()
    test_loss = 0
    for lang, d in test_data:
        torch.cuda.empty_cache()  # make sure the cache is emptied to begin the nexxt batch
        b_test = torch.from_numpy(d[:max_data, 0, :].astype("int32")).squeeze().to(device)
        if max_seq_len > 0:
            b_upos = torch.from_numpy(d[:max_data, 1, :max_seq_len].astype("int32")).squeeze().to(device).int()  # 
            b_deprel = torch.from_numpy(d[:max_data, 2, :max_seq_len].astype("int32")).squeeze().to(device).int()  # 
        else:
            b_upos = torch.from_numpy(d[:max_data, 1, :].astype("int32")).squeeze().to(device).int()  # 
            b_deprel = torch.from_numpy(d[:max_data, 2, :].astype("int32")).squeeze().to(device).int()  # 
        upos, deprel = model(b_test)
        # loss = loss_function(upos.view([-1, 18]), b_upos.view([-1]))
        # print("test tensor shapes: ", b_test.shape, upos.shape, b_upos.shape, deprel.shape, b_deprel.shape)
        loss, upos_loss, deprel_loss = loss_function(upos.view([-1, 18]), deprel.view([-1, 278]), b_upos.view([-1]), b_deprel.view([-1]))
        test_loss += loss.data.item()
        writer.add_scalar("LangLoss/test/"+lang, loss.data.item(), global_step=epoch)
        writer.add_scalar("LangLoss/test/upos/"+lang, upos_loss.data.item(), global_step=epoch)
        writer.add_scalar("LangLoss/test/deprel/"+lang, deprel_loss.data.item(), global_step=epoch)
        del b_test
        del b_upos
        del b_deprel
        torch.cuda.empty_cache()
    test_loss /= len(test_data)  # although this is not faire as different languages give different results
    writer.add_scalar("EpochLangLoss/test/", test_loss, global_step=epoch)
    print('epoch: {}====> Test set loss: {:.8f}'.format(epoch, test_loss))


def test_accuracy(model, test_data, epoch, device, max_data=50):
    torch.cuda.empty_cache()  # make sure the cache is emptied
    model.eval()
    epoch_acc = 0

    upos_eye = torch.eye(len(UPOS))
    deprel_eye = torch.eye(len(DEPREL))
    with torch.no_grad():
        upos_emb = nn.Embedding(*upos_eye.shape)
        upos_emb.weight.data.copy_(upos_eye)
        upos_emb = upos_emb.to(device)

        deprel_emb = nn.Embedding(*deprel_eye.shape)
        deprel_emb.weight.data.copy_(deprel_eye)
        deprel_emb.to(device)

    for lang, d in test_data:
        with torch.no_grad():
            b_test = torch.from_numpy(d[:max_data, 0, :].astype("int32")).squeeze().to(device)
            # TODO move the testing part to CPU so it takes less memory in the GPU and can keep training while testing
            # doing operations in boolean form so it takes less space in gpu
            b_upos = torch.from_numpy(d[:max_data, 1, :].astype("bool")).squeeze().to(device)
            b_deprel = torch.from_numpy(d[:max_data, 2, :].astype("bool")).squeeze().to(device)
            _, _, _, dec = model(b_test)
            #         last_latent = latent[-1]
            upos, deprel = dec
            ones = torch.ones(1).to(device)
            zeros = torch.zeros(1).to(device)
            upos = torch.where(upos > 0.9, ones, zeros).bool().to(device)
            deprel = torch.where(deprel > 0.9, ones, zeros).bool().to(device)
            upos = upos.view([-1, 18])
            deprel = deprel.view([-1, 278])

        # FIXME this accuracy measurement does not work.
        upos_acc = (upos == upos_emb(b_upos).view([-1, 18])).sum().item() / upos.shape[0]
        deprel_acc = (deprel == deprel_emb(b_deprel).view([-1, 278])).sum().item() / deprel.shape[0]
        acc = (upos_acc + deprel_acc) / 2
        # print("accuracy : ", acc, upos_acc, deprel_acc)
        writer.add_scalar("LangAccuracy/test/" + lang, acc, global_step=epoch)
        writer.add_scalar("LangAccuracy/test/upos/" + lang, upos_acc, global_step=epoch)
        writer.add_scalar("LangAccuracy/test/deprel/" + lang, deprel_acc, global_step=epoch)

        del b_test
        del b_upos
        del b_deprel
        torch.cuda.empty_cache()
    epoch_acc /= len(test_data)  # although this is not faire as different languages give different results
    writer.add_scalar("EpochLangAccuracy/test/", epoch_acc, global_step=epoch)
    print('epoch: {}====> Test Accuracy set loss: {:.4f}'.format(epoch, acc))
    pass


def load_test_data(base_dir, max_samples=-1, max_seq_len=-1):
    """finds all ud-treebank data that was pre-processed and saved in numpy and loads it.
    Each file is loaded and kept in a tuple (lang, dataset) and returns a list of those values
    if max_samples or max_seq_len are set to a nunmber greater than zero these will limit the data returned
    """
    # load testing data ALL the training data

    # get all file paths for testing
    all_fnames = get_all_files_recurse(base_dir)
    fnames = [f for f in all_fnames if "test-charse" in f and f.endswith(".npy")]
    # load all test files
    test_data = []
    for f in fnames:
        data = np.load(f)
        # data is shape: [total samples, data channels (3), 1024]
        # print("data loading shape: ", data.shape)
        if max_seq_len > 0:
            data = data[:, :, :max_seq_len]
        if max_samples > 0:
            data = data[:max_samples, :, :]
        lang_name = path_leaf(f).split("-ud")[0]
        test_data.append((lang_name, data))
    return test_data

