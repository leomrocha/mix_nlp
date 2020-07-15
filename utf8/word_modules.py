import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class SkipZero(nn.Module):
    """
    Applies the ReZero Technique to the skip connection
    """
    def __init__(self, module, use_res_weight=False):
        """
        :param module: Module to apply the skip connection to
        :param use_res_weight: if should also apply a weighting factor to the residual connection
        """
        super(SkipZero, self).__init__()
        self.module = module
        self.mod_weight = nn.Parameter(torch.Tensor([0.]))
        self.use_res_weight = use_res_weight
        if use_res_weight:
            self.res_weight = nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        res = x
        if self.use_res_weight:
            res = res * self.res_weight
        x = self.module(x) * self.mod_weight
        x = x + res
        return x


# Inspired in Fourier Series Representations, previous studies and the following paper
# SIREN https://arxiv.org/abs/2006.09661
# https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
class FourierActivation(nn.Module):
    """
    Fourier Activation, applies multiple (learned) weighted activation functions to the input
    This module should create an activation function approximated by Fourier Series
    """
    def __init__(self, n_coeff=10):
        """
        :param n_coeff: number of Fourier coefficients to use, the bigger it is the smoother the final activation function
        """
        super(FourierActivation, self).__init__()

        sines = np.linspace(0., 2*np.pi-0.00001, n_coeff)  # linspace is inclusive of last point ... hence -0.00001
        self.wx = nn.Parameter(torch.ones(len(sines))).unsqueeze(0).transpose(0, 1)  # input vector scaling
        self.phases = nn.Parameter(torch.Tensor(sines)).unsqueeze(0).transpose(0, 1)  # phase shifts
        self.amplitud = nn.Parameter(torch.ones(len(sines))).unsqueeze(0).transpose(0, 1)  # amplitud scaling -> weight

        # TODO should do some other initialization technique here instead of ones and fixed values

    def forward(self, x):
        # compute fourier coefficients
        coefs = self.amplitud * torch.cos(self.wx * x - self.phases)
        # sum all values per column
        res = torch.sum(coefs, 0)

        return res


class WordEncoder(nn.Module):
    def __init__(self,
                 in_lang_size, hid_lang_size, lang_mat_size,
                 in_word_size, hid_word_size, word_mat_size,
                 activation=None,
                 ):
        super(WordEncoder, self).__init__()
        # language FF encoding
        ff_lang = nn.Sequential(
            nn.Linear(in_lang_size, hid_lang_size),
            nn.Linear(hid_lang_size, lang_mat_size),
        )
        self.ff_lang = SkipZero(ff_lang)
        # language activation
        self.lang_activ = FourierActivation() if activation is None else activation
        # language encoding matrix

        # word vector FF encoding

        # word activation

        # composed FF encoding

        # composed activation

    def froze_embed_train(self, mode=True):
        super().train(mode)
        # self.embeds.requires_grad_(False)

    def forward(self, lang_vec, word_vec):
        """
        :param lang_vec: original language input
        :param word_vec:
        :return: latent
        """

        return latent

