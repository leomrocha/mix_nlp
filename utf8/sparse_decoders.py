"""
Pytorch Decoders modules generators for multi-hot encodings

Several different techniques are used in order to test them,
"""

import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np


class UTF8SparseDecoderModule(nn.Module):
    """
    Decoder Module for UTF8 coding where the code is generate by a redundant code with
    """

    # def __init__(self, input_size=192, segments=3, N=37, k=13, coprimes=(11, 13, 19, 23), cycles=(11, 7, 4, 3),
    #              use_transformer=True):
    def __init__(self, input_size=128, segments=2, N=24, k=3, coprimes=(3, 5, 11, 13), cycles=(6, 2),
                 use_transformer=True, transformer_ff_size=1024):

        """"""
        super(UTF8SparseDecoderModule, self).__init__()
        self.segments = segments
        self.use_transformer = use_transformer
        # input passes through linear to get to the final dimension
        self._code_dim = N + np.sum(coprimes) + np.sum(cycles)
        hidd_size = max(input_size * 4, self._code_dim * 4)
        self.linear = nn.Sequential(
            weight_norm(nn.Linear(input_size, hidd_size)),
            weight_norm(nn.Linear(hidd_size, self._code_dim))
        )
        if use_transformer:
            nheads = k + len(coprimes) + len(cycles)
            self.transformer = TransformerEncoderLayer(self._code_dim, nheads, dim_feedforward=transformer_ff_size)

    def forward(self, x):
        # input x must be of shape:
        # [batch size, sequence length, embedding] ???? TODO fix this
        y = self.linear(x)
        if self.use_transformer:
            y = self.transformer(y)
        y = F.sigmoid(y)
        return y


class UTF8SegmentMultihotDecoderModule(nn.Module):
    """
    Decoder Module for Hand Designed for Per Segment Multi-One-Hot Coding

    """
    @classmethod
    def get_lin_size(cls, segments):
        """

        :param segments: number of segments in the code
        :return: the linear layer size for the code

        """
        size_dict = {
            1: 260,
            2: 324,
            3: 388,
            4: 452
        }
        return size_dict[segments]

    def __init__(self, input_size, hidd_size=1024, segments=2, use_transformer=False, dropout=0.1):
        """
        :param input_size: vector size (dimensions) of the input
        :param segments: segments used in the code -> defines the size of the output
        :param use_transformer: if a transformer is used in the decoder layer
        :param use_softmax: if softmax is used at the end of the decoder, multiple softmax will be used
        (one per segment + one for the 4 dimensional segment coding at the beginning)
        """
        super(UTF8SegmentMultihotDecoderModule, self).__init__()
        self.segments = segments
        self.use_transformer = use_transformer
        lin_size = self.get_lin_size(segments)
        self.linear = nn.Sequential(
            weight_norm(nn.Linear(input_size, hidd_size)),
            weight_norm(nn.Linear(hidd_size, lin_size))
        )
        if use_transformer:
            # number of heads in the transformer is the segments plus the segment coding part
            nheads = segments + 1
            self.transformer = TransformerEncoderLayer(lin_size, nheads, dim_feedforward=lin_size*4, dropout=dropout)
        self.lin_segment = weight_norm(nn.Linear(lin_size, 4))
        self.lin_segment_1 = weight_norm(nn.Linear(lin_size, 256))
        if segments > 1:
            self.lin_segment_2 = weight_norm(nn.Linear(lin_size, 64))
        if segments > 2:
            self.lin_segment_3 = weight_norm(nn.Linear(lin_size, 64))
        if segments > 3:
            self.lin_segment_4 = weight_norm(nn.Linear(lin_size, 64))

    def forward(self, x):
        # input x must be of shape:
        # [batch size, sequence length, embedding] ???? TODO fix this
        y = self.linear(x)
        if self.use_transformer:
            y = self.transformer(y)
        out_seg = [self.lin_segment(y), F.softmax(self.lin_segment_1(y))]
        if self.segments > 1:
            out_seg.append(F.softmax(self.lin_segment_2(y)))
        if self.segments > 2:
            out_seg.append(F.softmax(self.lin_segment_3(y)))
        if self.segments > 3:
            out_seg.append(F.softmax(self.lin_segment_4(y)))
        out = torch.cat(out_seg, dim=-1)  # TODO check dimension -> depends on the shape of the input
        return out
