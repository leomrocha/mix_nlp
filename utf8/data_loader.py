"""

Loading Data


# TODO
Do the Data Loading with NVIDIA DALI instead, examples and documentation here:

https://github.com/NVIDIA/DALI
https://developer.nvidia.com/DALI
https://docs.nvidia.com/deeplearning/sdk/dali-developer-guide/docs/index.html
https://github.com/NVIDIA/DALI/tree/master/docs/examples

https://towardsdatascience.com/nvidia-dali-speeding-up-pytorch-876c80182440
https://github.com/yaysummeriscoming/DALI_pytorch_demo

"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import IterableDataset

try:
    from .constants import *
except:
    # hack to solve issue with ipython executing this import
    from constants import *


class Txt2TxtDataset(IterableDataset):
    def __init__(self, root_dir, codedict, separator='█', special_codes=SPECIAL_CODES):
        self.root_dir = root_dir
        self.separator = separator
        self.codedict = codedict
        self.special_codes = special_codes
        # verify that the special codes are present
        for c in special_codes:
            if c[0] not in self.codedict:
                self.codedict[c[0]] = c[1]

    def _txt2tensor(self, txt):
        return torch.from_numpy(np.array(map(self._item2int, txt)))

    def _item2int(self, char):
        if char not in self.codedict:
            char = self.unk
        num = self.codedict[char]
        return num

    def process_line(self, line):
        # Splits the line into text and label and applies preprocessing to the text
        items = line.split(self.separator)
        out_items = map(self._item2int, items)
        return out_items

    def __iter__(self):
        iterator = open(self.root_dir, 'rb')
        ret = map(self.process_line, iterator)
        return ret

    # def __next__(self):
    #     # .... ???
    #     if :
    #         return x
    #     else:
    #         raise StopIteration

