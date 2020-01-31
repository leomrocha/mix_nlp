from datetime import datetime
import numpy as np
from multiprocessing import Pool, cpu_count
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .models import *

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

