import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

try:
    # from .utils import *
    from .model_blocks import *
    from .sparse_decoders import UTF8SparseDecoderModule
except:
    # to solve issue with ipython executing this import
    # from utils import *
    from model_blocks import *
    from sparse_decoders import UTF8SparseDecoderModule


class ConvModel(nn.Module):
    def __init__(self, embed_matrix,
                 seq_len=512, lang_seq_len=60,
                 in_dim=96, hidd_dim=1024, cnv_dim=192,  # linear layers at the input for embedding projections
                 c_in=(192, 512, 1024, 1024, 512, 192), c_out=(512, 1024, 1024, 512, 192, 192),  # channels for blocks
                 b_layers=(3, 5, 5, 5, 3),  # number of layers for each bloc
                 first_k_size=3, kernel_size=3, cnv_dropout=0.3, groups=4, cnv_activation="gelu",

                 dec_input_size=192, segments=2, N=24, k=3, coprimes=(3, 5, 11, 13), cycles=(4, 6, 8, 10, 12),
                 dec_use_transformer=True, transformer_ff_size=1024, dec_activation='gelu', dec_dropout=0.1
                 ):

        super(ConvModel, self).__init__()
        self.seq_len = seq_len
        self.lang_seq_len = lang_seq_len
        # needs the Embedding matrix
        # with torch.no_grad():
        self.embeds = nn.Embedding(*embed_matrix.shape)
        self.embeds.weight.data.copy_(torch.from_numpy(embed_matrix))
        self.embeds.requires_grad = False
        # Input projection of the embedding
        self.lin = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hidd_dim)),
            weight_norm(nn.Linear(hidd_dim, cnv_dim)),
        )
        # Convolutional Column
        self.conv = ConvColumn(c_in, c_out, b_layers, first_k_size, kernel_size, cnv_dropout, groups, cnv_activation)
        # # decoder layers
        # Decoding separation in sequence length for language detection AND language model  # TODO pass params instead
        self.lang_lin = weight_norm(nn.Linear(seq_len, lang_seq_len))
        # self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, dim_feedforward=1024,
        #                                                activation='gelu', dropout=dec_dropout)
        self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, activation='gelu')

        self.lm_lin = weight_norm(nn.Linear(seq_len, seq_len))  # this is just a linear transformation
        self.lm_transformer = TransformerEncoderLayer(seq_len, 8, 2048, activation='gelu')

        # Embedding projection decoding
        # TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu")
        self.decoder = UTF8SparseDecoderModule(dec_input_size, segments, N, k, coprimes, cycles, dec_use_transformer,
                                               transformer_ff_size, dec_activation, dec_dropout)
        # # needs the FAISS decoder -> TODO implement and send to GPU, FIXME faiss failing to load in gpu
        # self.indexl2 = faiss.IndexFlatL2(embed_matrix.shape[1])
        # self.indexl2.add(embed_matrix)

    def forward(self, x_in, decode_faiss=False):
        # [batch, sequence (long)]
        # Embedding
        x = self.embeds(x_in)
        # [batch, sequence, embed]
        # Now adapt for convolutions
        x = self.lin(x)
        # work in sequence (time) dimension
        x = x.transpose(1, 2)
        # [batch, embed, sequence]
        x = self.conv(x)

        # Apply the transformer layers here
        x_lang = self.lang_lin(x)
        # x_lang = self.lang_transformer(x_lang)
        x_lm = self.lm_lin(x)
        # x_lm = self.lm_transformer(x_lm)
        # concatenate over the sequence
        x = torch.cat([x_lm, x_lang], dim=-1)

        # Go back to space dimension only (character level) for decoder
        x = x.transpose(1, 2)
        # [batch, sequence, embed]
        x = self.decoder(x)
        x_lang = x[:, self.seq_len:, :]
        x_lm = x[:, :self.seq_len, :]
        # if decode_faiss:
        #     k = 1
        #     D, I = self.indexl2.search(x.view(-1, x.shape[-1]), k)
        #     return I.view(* x.shape[:-1])
        return x_lm, x_lang

