"""
Helper functions for training and testing, for the moment many things contain absolute paths and is quite dirty
The thing is, I'll be improving the code as the research advances
"""

import os
import sys
import numpy as np
from datetime import datetime
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from torch.utils.checkpoint import *
# for gradient checkpointing (doesn't seems so straightforward) check:
# https://github.com/prigoyal/pytorch_memonger/blob/master/tutorial/Checkpointing_for_PyTorch_models.ipynb
# https://github.com/pytorch/pytorch/pull/4594
from torch.utils.tensorboard import SummaryWriter

from apex import amp, optimizers

try:
    from .data_loader import *
except:
    # Ugly hack for Jupyter Lab not loading correctly
    from data_loader import *


# Gradient checkpointing .... TODO when/if necessary


# from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(data, n, dim=0):
    """Yield successive n-sized chunks from data by the dimension dim"""
    for i in range(0, data.shape[dim], n):
        yield data[i:i + n, :, :]


def loss_txt2txt_multi(prediction, target,
                       pred_dest_lang, tgt_dest_lang,
                       scale_loss=1.,
                       scale_dest_lang_loss=1.,
                       fn_loss=F.nll_loss,  # fn_loss=F.kl_div
                       ):
    """"""
    pred_loss = fn_loss(prediction, target) * scale_loss
    dest_lang_loss = fn_loss(pred_dest_lang, tgt_dest_lang) * scale_dest_lang_loss
    loss = pred_loss + dest_lang_loss
    return loss, (pred_loss, dest_lang_loss)


def main(model, train_files, test_files, codebook_file,
         batch_size=10000, num_workers=10, max_seq_len=512, add_noise_to_task=True,
         optimizer='FusedAdam',
         lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
         amsgrad=False, adam_w_mode=True, max_grad_norm=1.0):
    # TODO if CUDA not available, ... something should be done
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

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

    f = open(codebook_file, 'rb')
    codebook, char2int, int2char = pickle.load(f)
    train_dataset = Txt2TxtDataset(train_files, char2int, max_len=max_seq_len, add_noise_to_task=add_noise_to_task)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,
                                   pin_memory=True,
                                   num_workers=num_workers, worker_init_fn=Txt2TxtDataset.worker_init_fn)
    test_dataset = Txt2TxtDataset(test_files, char2int, max_len=max_seq_len, add_noise_to_task=add_noise_to_task)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,
                                  pin_memory=True,
                                  num_workers=num_workers, worker_init_fn=Txt2TxtDataset.worker_init_fn)

    criterion = loss_txt2txt_multi
    train_main(model, optimizer, train_data_loader, test_data_loader, criterion, noise_in_task=add_noise_to_task)


def train_main(model, optimizer, train_data_loader, test_data_loader,
               checkpoint_path,  # where to save the checkpoints
               criterion=loss_txt2txt_multi,
               noise_in_task=False, opt_level="O1",
               test_period=10
               ):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    model.train()
    # if device == 'cuda:0':  # for later to make sure things work in cpu too
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level,  # [O0, O1, O2, O3 ]

                                          )
    writer = SummaryWriter()
    batch_count = 1
    epoch_count = 1

    for train_batch_data in train_data_loader:
        train_batch(model, optimizer, criterion, train_batch_data, batch_count, writer)
        if batch_count % test_period == 0:
            model.eval()
            # test on one batch ...
            try:
                test_batch_data = test_data_loader.next()
                test_batch(model, criterion, test_batch_data, epoch_count, writer)
                # update counters and make model trainable again
            except StopIteration as e:
                print("No more batches to test ... {}".format(e))
                pass
            epoch_count += 1
            model.train()
            # save checkpoint of the model
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            dtime = datetime.now().isoformat()
            cp_name = os.path.join("amp-checkpoint_opt-{}_{}.pt".format(opt_level, dtime))
            torch.save(checkpoint, cp_name)
        batch_count += 1


def train_batch(model, optimizer, criterion, batch_data, batch_count, summary_writer):
    batch_len = len(batch_data)
    if batch_len == 3:
        source, target, target_lang = batch_data
    elif batch_len == 5:
        noise_masked, noise_target, source, target, target_lang = batch_data
    else:
        raise NotImplementedError("input batch must have either 3 or 5 elements")
    # train in the source, target, target_lang tuple
    optimizer.zero_grad()
    out = model(source)
    pred, pred_lang = out
    loss, ind_losses = criterion(pred, source, pred_lang, target_lang)

    # Write summaries and details on TensorBoard for the task
    names = ['task_loss', 'lang_det_loss']
    summary_writer.add_scalar("Loss/train", loss.data.item(), global_step=batch_count)
    for n, l in zip(names, ind_losses):
        ldata = l.data.item()
        summary_writer.add_scalar("Loss/train-{}".format(n), ldata)
    with amp.scaled_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    optimizer.step()

    if batch_len == 5:
        # also train in the noise_masked, noise_target, target_lang tuple
        optimizer.zero_grad()
        out = model(noise_masked)
        pred, pred_lang = out
        loss, ind_losses = criterion(pred, source, pred_lang, target_lang)

        # Write summaries and details on TensorBoard for the task
        names = ['lm_task_loss', 'lm_lang_det_loss']
        summary_writer.add_scalar("Loss/lm_train", loss.data.item(), global_step=batch_count)
        for n, l in zip(names, ind_losses):
            ldata = l.data.item()
            summary_writer.add_scalar("Loss/train-{}".format(n), ldata)
        with amp.scaled_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
    torch.cuda.empty_cache()


def test_batch(model, criterion, batch_data, batch_count, summary_writer):
    batch_len = len(batch_data)
    if batch_len == 3:
        source, target, target_lang = batch_data
    elif batch_len == 5:
        noise_masked, noise_target, source, target, target_lang = batch_data
    else:
        raise NotImplementedError("input batch must have either 3 or 5 elements")
    # train in the source, target, target_lang tuple
    out = model(source)
    pred, pred_lang = out
    loss, ind_losses = criterion(pred, source, pred_lang, target_lang)

    # Write summaries and details on TensorBoard for the task
    names = ['task_loss', 'lang_det_loss']
    summary_writer.add_scalar("Loss/test", loss.data.item(), global_step=batch_count)
    for n, l in zip(names, ind_losses):
        ldata = l.data.item()
        summary_writer.add_scalar("Loss/test-{}".format(n), ldata)

    if batch_len == 5:
        out = model(noise_masked)
        pred, pred_lang = out
        loss, ind_losses = criterion(pred, source, pred_lang, target_lang)

        # Write summaries and details on TensorBoard for the task
        names = ['lm_task_loss', 'lm_lang_det_loss']
        summary_writer.add_scalar("Loss/lm_test", loss.data.item(), global_step=batch_count)
        for n, l in zip(names, ind_losses):
            ldata = l.data.item()
            summary_writer.add_scalar("Loss/test-{}".format(n), ldata)
    torch.cuda.empty_cache()


