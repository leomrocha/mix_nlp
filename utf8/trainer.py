from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pickle
import os
try:
    from .models import *
    from .trainer_helpers import *
    from .data_loader import *
    from .utils import *
    from .tools import *
except:
    from models import *
    from trainer_helpers import *
    from data_loader import *
    from utils import *
    from tools import *



# TODO put this in a config file
fcodebook = "/home/leo/projects/minibrain/predictors/sequence/text/utf8-codes/utf8_codebook_overfit_matrix_2seg_dim64.npy"
utf8codematrix = "/home/leo/projects/minibrain/predictors/sequence/text/utf8-codes/utf8_code_matrix_2seg.npy"
dataset_train = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.5/traindev_np_batches_779000x3x1024_uint16.npy"
BASE_DATA_DIR_UD_TREEBANK = "/home/leo/projects/Datasets/text/UniversalDependencies/ud-treebanks-v2.5"

# cuda seems to reverse the GPU ids with CUDA id so ... mess
# Cuda maps cuda:0 to my RTX 2080ti (GPU#1) and
# Cuda maps cuda:1 to my GTX 1080 (GPU#0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

BASE_PATH = '/home/leo/projects/Datasets/text/selected_monofile/partitions'
codebook_path = '/home/leo/projects/mix_nlp/utf8/codes/adhoc-codebook-1871.pkl'

CHKP_PATH = "/media/nfs/mix_nlp/checkpoints"
# chkp_fname = os.path.join(chkp_path, "amp-checkpoint_opt-O2_loss-0.001_2020-02-15T11:12:26.762745.pt")
# chkp_fname = os.path.join(chkp_path, "amp-checkpoint_opt-O2_loss-0.003_2020-02-17T13:12:18.881112.pt")
# chkp_fname = os.path.join(chkp_path, "amp-checkpoint_opt-O2_loss-2.222_2020-02-17T15:39:56.682146.pt")
# CHKP_FNAME = os.path.join(CHKP_PATH, "amp-checkpoint_opt-O2_loss-0.539_2020-02-18T00:38:03.332333.pt")
CHKP_FNAME = os.path.join(CHKP_PATH, "amp-checkpoint_opt-O2_batch-1000_loss-0.005_2020-02-18T10:59:32.156309.pt")


def load_checkpoint(clean_model, fname, optimizer=None, amp=None):
    chkp = torch.load(fname)
    clean_model.load_state_dict(chkp['model'])
    if optimizer:
        optimizer.load_state_dict(chkp['optimizer'])
    if amp:
        amp.load_state_dict(chkp['amp'])
    return chkp


def preload_model(model, fname, embed_matrix):
    # model state is modified inside
    chkp = load_checkpoint(model, fname)
    # Make sure the embeddings are the ones I had set manually
    model.embeds.weight.data.copy_(torch.from_numpy(embed_matrix))
    # self.embeds.requires_grad = False
    model.embeds.requires_grad_(False)
    print("Set model embeds to original binary ones, requires_grad={}".format(model.embeds.weight.requires_grad))


def train(ModelClass=None, chkp_fname=None):
    fpaths = get_all_files_recurse(BASE_PATH)

    train_files = [f for f in fpaths if 'train' in f]
    dev_files = [f for f in fpaths if 'dev' in f]
    valid_files = [f for f in fpaths if 'valid' in f]

    train_glue_files = [f for f in train_files if 'glue-' in f]
    dev_glue_files = [f for f in dev_files if 'glue-' in f]

    train_all_files = [f for f in train_files if 'all' in f]
    test_all_files = [f for f in dev_files if 'all' in f]

    train_files = [f for f in train_files if 'glue' not in f and 'all' not in f]
    test_files = [f for f in dev_files if 'glue' not in f and 'all' not in f]

    f = open(codebook_path, 'rb')
    codebook, char2int, int2char = pickle.load(f)
    if ModelClass:
        model = ModelClass(codebook)
    else:
        model = ConvModel(codebook)
    # ## WARNING
    # hiatus, this is to start from the pretrained so it's faster for training later
    # but I'll modify again the embeddings to the ones I had set before
    if chkp_fname:
        preload_model(model, chkp_fname, codebook)
    # ## END WARNING

    print("Total Params: {} | Trainable params: {} ".format(count_parameters(model), count_trainable_parameters(model)))
    model = model.to(device)

    trainer_helper_main(model, train_files, test_files, codebook_path,
                        #      batch_size=10,
                        #      batch_size=175, # with opt_level=O1 this is the max
                        batch_size=185,  # this one works with opt_level=O2
                        # optimizer='FusedAdam',  # Adam goes down really fast but then starts giving losses as NaN
                        optimizer='FusedLAMB',
                        # Fused lamb decreases slowly but steady and goes to better loss than Adam.
                        # NaN after 21730 batches, 13h30m36s
                        #     optimizer='FusedNovoGrad', # is definitely the slowest one at the beginning,
                        #     stabilizes at the worst value
                        opt_level='O2',
                        # add_noise_to_task=False,
                        add_noise_to_task=True,
                        add_str_noise_to_input=True,
                        test_period=-1,  # No tests, as I don't know why they are not called ... FIXME
                        #      checkpoint_period=50,
                        checkpoint_period=100,
                        checkpoint_path="/media/nfs/mix_nlp/checkpoints"
                        )


if __name__ == "__main__":
    print("STARTING TRAINING")
    # train(ConvModel, chkp_fname=CHKP_FNAME)
    #
    train(ConvModel)
    # train()
