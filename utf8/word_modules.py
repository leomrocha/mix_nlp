import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class SkipZero(nn.Module):
    """
    Applies the ReZero Technique to the skip connection,
    uses projection for modules where input and output shapes are different
    """
    def __init__(self, module, in_size=None, out_size=None, use_res_weight=False):
        """
        :param module: Module to apply the skip connection to
        :param in_size: input size, leave blank if input and output size are the same
        :param out_size: output size, leave blank if input and output size are the same
        :param use_res_weight: if should also apply a weighting factor to the residual connection
        """
        super(SkipZero, self).__init__()
        self.module = module
        self.mod_weight = nn.Parameter(torch.Tensor([0.]))
        self.proj = None
        if in_size and out_size and (in_size != out_size):
            self.proj = nn.Linear(in_size, out_size)
        self.use_res_weight = use_res_weight
        if use_res_weight:
            self.res_weight = nn.Parameter(torch.Tensor([1.]))

    def forward(self, x):
        res = x
        if self.proj:
            res = self.proj(res)
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
        self.wx = nn.Parameter(torch.ones(len(sines))).unsqueeze(1)  # input vector scaling
        self.phases = nn.Parameter(torch.Tensor(sines)).unsqueeze(1)  # phase shifts
        self.amplitud = nn.Parameter(torch.ones(len(sines))).unsqueeze(1)  # amplitud scaling -> weight

        # TODO should do some other initialization technique here instead of ones and fixed values

    def forward(self, x):
        # compute fourier coefficients
        coefs = self.amplitud * torch.cos(self.wx * x - self.phases)
        # sum all values per column  # TODO FIXME fix the sum dimension
        res = torch.sum(coefs, dim=0)

        return res


class WordEncoder(nn.Module):
    def __init__(self,
                 in_lang_size, in_word_size,
                 hid_lang_size=2048, hid_word_size=2048, lat_word_size=300,
                 activation=None,
                 use_softmax=False,
                 ):
        super(WordEncoder, self).__init__()
        self.use_softmax = use_softmax
        self.in_lang_size = in_lang_size
        # language FF encoding
        ff_lang = nn.Sequential(
            nn.Linear(in_lang_size, hid_lang_size),
            nn.Linear(hid_lang_size, lat_word_size),
        )
        self.ff_lang = SkipZero(ff_lang, in_lang_size, lat_word_size)
        # language activation
        self.lang_activ = FourierActivation() if activation is None else activation
        # language encoding matrix
        self.lang_gate_matrix = nn.Parameter(
            torch.rand(lat_word_size, lat_word_size))  # TODO make a better initialization
        # word vector FF encoding
        ff_word = nn.Sequential(
            nn.Linear(in_word_size, hid_word_size),
            # nn.Linear(hid_word_size, hid_word_size),
            nn.Linear(hid_word_size, lat_word_size),
        )
        self.ff_word = SkipZero(ff_word, in_word_size, lat_word_size)
        # word activation
        self.word_activ = FourierActivation() if activation is None else activation

        # composed FF encoding
        ff_compose = nn.Sequential(
            nn.Linear(lat_word_size, hid_word_size),
            nn.Linear(hid_word_size, lat_word_size),
        )
        self.ff_compose = SkipZero(ff_compose)
        # composed activation
        self.compose_activ = FourierActivation() if activation is None else activation

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.lang_gate_matrix)

    def forward(self, word_vec, lang_vec=None):
        """
        :param lang_vec: original language input
        :param word_vec:
        :return: latent
        """
        if not lang_vec:
            # TODO FIX for batch sizes too
            lang_vec = torch.zeros(self.in_lang_size)
        lat_lang = self.lang_activ(self.ff_lang(lang_vec))
        # word_gate = torch.mm(lat_lang, self.lang_matrix)
        # Compute attention and gate the language matrix
        # if self.use_softmax:
        #     lat_lang = F.softmax(lat_lang)  # soft attention over language ... ? I don't think it'll work here
        # TODO FIX the transpose and dot product order here
        word_gate_matrix = lat_lang * self.lang_gate_matrix  # this matrix should map the language
        if self.use_softmax:
            word_gate_matrix = F.softmax(word_gate_matrix, dim=-1)  # soft attention per column, not much faith in this

        lat_word = self.word_activ(self.ff_word(word_vec))

        lat_word = torch.mm(lat_word, word_gate_matrix)

        latent = self.compose_activ(self.ff_compose(lat_word))

        return latent


class WordDecoder(nn.Module):
    def __init__(self,
                 in_lang_size, out_langvar_size, out_word_size,
                 hid_lang_size=2048, hid_word_size=2048, lat_word_size=300,
                 activation=None,
                 # use_softmax=False,
                 ):
        super(WordDecoder, self).__init__()
        # self.use_softmax = use_softmax
        # out_lang_size, out_langvar_size, out_word_size,

        # latent
        ff_latent = nn.Sequential(
            nn.Linear(lat_word_size, hid_word_size),
            # nn.Linear(hid_word_size, hid_word_size),
            nn.Linear(hid_word_size, lat_word_size),
        )
        self.ff_latent = SkipZero(ff_latent)
        self.ff_latent_activ = FourierActivation() if activation is None else activation

        # language and langvar detection
        lat_lang_detect_size = in_lang_size + out_langvar_size
        ff_lang_decode = nn.Sequential(
            nn.Linear(lat_word_size, hid_lang_size),
            nn.Linear(hid_lang_size, lat_lang_detect_size),
        )
        self.ff_lang_decode = SkipZero(ff_lang_decode)

        self.ff_lang_detect = nn.Linear(lat_lang_detect_size, in_lang_size)
        self.ff_lang_detect_activ = FourierActivation() if activation is None else activation

        self.ff_langvar_detect = nn.Linear(lat_lang_detect_size, out_langvar_size)
        self.ff_langvar_detect_activ = FourierActivation() if activation is None else activation

        # output language encoding
        ff_lang = nn.Sequential(
            nn.Linear(in_lang_size, hid_lang_size),
            nn.Linear(hid_lang_size, lat_word_size),
        )
        self.ff_lang = SkipZero(ff_lang, in_lang_size, lat_word_size)
        # language activation
        self.lang_activ = FourierActivation() if activation is None else activation
        # language encoding matrix
        self.lang_gate_matrix = nn.Parameter(torch.rand(lat_word_size, lat_word_size))

        # output word decoding

        ff_out_word = nn.Sequential(
            nn.Linear(lat_word_size, hid_word_size),
            nn.Linear(hid_word_size, out_word_size),
        )
        self.ff_out_word = SkipZero(ff_out_word)
        # composed activation
        self.out_word_act = FourierActivation() if activation is None else activation

    def init_weights(self):
        torch.nn.init.xavier_normal_(self.lang_gate_matrix)

    def forward(self, latent, out_lang_code):
        """

        :param latent: latent
        :param out_lang_code: output language code
        :return: lang_code, langvar_code, word_code
          (detected input language, detected variation, output decoded word)
        """
        latent = self.ff_latent(latent)
        latent = self.ff_latent_activ(latent)

        # branch for language detection
        lat_lang = self.ff_lang_decode(latent)

        lang_code = self.ff_lang_detect(lat_lang)
        lang_code = self.ff_lang_detect_activ(lang_code)

        langvar_code = self.ff_langvar_detect(lat_lang)
        langvar_code = self.ff_langvar_detect_activ(langvar_code)

        # output language encoding
        lat_out_lang = self.ff_lang(out_lang_code)
        lat_out_lang = self.lang_activ(lat_out_lang)
        # TODO FIX the transpose and dot product order here
        word_gate_matrix = lat_out_lang * self.lang_gate_matrix  # this matrix should map the language
        # word decoding
        word_code = torch.mm(latent, word_gate_matrix)
        word_code = self.ff_out_word(word_code)
        word_code = self.out_word_act(word_code)

        return word_code, lang_code, langvar_code

# TODO all the tests ... ;)
