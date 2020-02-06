"""

Loading Data


# TODO
Loading Data with NVIDIA DALI, examples and documentation here:

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

# command codes of the start
# {"<start>": 0x02, "<end>": 0x03, "<unk>": 0x15})
# use Box Codes for the delimitation to avoid any encoding collition
NIL = ('◁nil▷', 0x00)
UNK = ('◁unk▷', 0x02)
SOT = ('◁sot▷', 0x03)  # start of text
EOT = ('◁eot▷', 0x15)  # end of text


class Txt2TxtDataset(IterableDataset):
    def __init__(self, root_dir, codedict, separator='█', nil=NIL, unk=UNK, sot=SOT, eot=EOT):
        self.root_dir = root_dir
        self.separator = separator
        self.codedict = codedict

        self.nil = nil[0]
        self.unk = unk[0]
        self.sot = sot[0]
        self.eot = eot[0]
        # verify that the right codes are present
        for c in [nil, unk, sot, eot]:
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
        # here must construct the sample for idx
        iterator = open(self.root_dir, 'rb')
        ret = map(self.process_line, iterator)
        return ret

    # def __next__(self):
    #     # .... ???
    #     if :
    #         return x
    #     else:
    #         raise StopIteration

