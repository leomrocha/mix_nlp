import faiss
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm
# from fairseq.modules.dynamic_convolution import DynamicConv
from fairseq.models.lightconv import LightConvEncoderLayer, base_architecture
from fairseq.modules.positional_embedding import PositionalEmbedding

import argparse

try:
    # from .utils import *
    from .model_blocks import *
    from .sparse_decoders import UTF8SparseDecoderModule
except:
    # to solve issue with ipython executing this import
    # from utils import *
    from model_blocks import *
    from sparse_decoders import UTF8SparseDecoderModule


class ReZeroTransformerModel(nn.Module):
    def __init__(self, d_model=512, n_head=8, n_hid=2048, n_layers=6, dropout=0.1, activation='relu'):
        super(ReZeroTransformerModel, self).__init__()

        self.embeds = nn.Embedding()
        # TODO here the code for deciding which variant to use, fixed embedding or NOT
        self.transformer = ReZeroTransformerModule(d_model=d_model, n_head=n_head, n_hid=n_hid,
                                         n_layers=n_layers, dropout=dropout, activation=activation)
        self.decoder =
        # TODO here do the decision on the variant to use for decoding:
        # one-hot, faiss

    def forward(self, x):
        ebd = self.embeds(x)
        code = self.transformer(ebd)
        dec = self.decoder(code)
        return dec


class ConvModel(nn.Module):
    def __init__(self, embed_matrix,
                 seq_len=512, lang_seq_len=60, vocab_size=1871,
                 in_dim=96, hidd_dim=1024, cnv_dim=192,  # linear layers at the input for embedding projections
                 c_in=(192, 512, 1024, 1024, 512, 192), c_out=(512, 1024, 1024, 512, 192, 192),  # channels for blocks
                 b_layers=(3, 5, 5, 5, 3),  # number of layers for each bloc
                 first_k_size=3, kernel_size=3, cnv_dropout=0.3, groups=4, cnv_activation="relu",

                 dec_input_size=192, segments=2, N=24, k=3, coprimes=(3, 5, 11, 13), cycles=(4, 6, 8, 10, 12),
                 dec_use_transformer=True, transformer_ff_size=1024, dec_activation='gelu', dec_dropout=0.1
                 ):

        super(ConvModel, self).__init__()
        self.seq_len = seq_len
        self.lang_seq_len = lang_seq_len
        self.vocab_size = vocab_size
        # needs the Embedding matrix
        # with torch.no_grad():
        self.embeds = nn.Embedding(*embed_matrix.shape)
        self.embeds.weight.data.copy_(torch.from_numpy(embed_matrix))
        # self.embeds.requires_grad = False
        self.embeds.requires_grad_(False)
        # self.embeds = self.embeds
        # Input projection of the embedding
        self.lin = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hidd_dim)),
            weight_norm(nn.Linear(hidd_dim, cnv_dim)),
        )
        # Convolutional Column
        self.conv = ConvColumn(c_in, c_out, b_layers, first_k_size, kernel_size, cnv_dropout, groups, cnv_activation)
        # # decoder layers
        # Decoding separation in sequence length for language detection AND language model
        self.lang_lin = weight_norm(nn.Linear(seq_len, lang_seq_len))
        # self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, dim_feedforward=1024,
        #                                                activation='gelu', dropout=dec_dropout)
        # self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, activation='gelu')
        lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, activation='gelu')
        self.lang_transformer = TransformerEncoder(lang_transformer, 3)

        self.lm_lin = weight_norm(nn.Linear(seq_len, seq_len))  # this is just a linear transformation
        # self.lm_transformer = TransformerEncoderLayer(seq_len, 8, 2048, activation='gelu')
        lm_transformer = TransformerEncoderLayer(seq_len, 8, 2048, activation='gelu')
        self.lm_transformer = TransformerEncoder(lm_transformer, 3)
        # Embedding projection decoding
        # TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu")
        self.decoder = UTF8SparseDecoderModule(dec_input_size, segments, N, k, coprimes, cycles, dec_use_transformer,
                                               transformer_ff_size, dec_activation, dec_dropout)
        # # needs the FAISS decoder -> TODO implement and send to GPU
        # self.indexl2 = faiss.IndexFlatL2(embed_matrix.shape[1])
        # self.indexl2.add(embed_matrix)
        self.sfmx_lin = weight_norm(nn.Linear(in_dim, vocab_size))
        # self.softmax = nn.LogSoftmax(dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self.embeds.requires_grad_(False)

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
        x = x.transpose(1, 2).contiguous()
        # [batch, sequence, embed]
        x = self.decoder(x)
        # mapping to dimension for softmax layer
        x = self.sfmx_lin(x)
        # decoding (should change for FAISS instead when I make it work correctly
        # x = self.softmax(x)
        x = F.log_softmax(x, dim=-1)
        x_lang = x[:, self.seq_len:, :].contiguous()
        x_lm = x[:, :self.seq_len, :].contiguous()
        # if decode_faiss:
        #     k = 1
        #     D, I = self.indexl2.search(x.view(-1, x.shape[-1]), k)
        #     return I.view(* x.shape[:-1])
        # print(self.seq_len, self.lang_seq_len, x.shape, x_lm.shape, x_lang.shape)
        return x_lm, x_lang


class DynConvColModel(nn.Module):
    def __init__(self, embed_matrix,
                 seq_len=512, lang_seq_len=60, vocab_size=1871,
                 in_dim=96, hidd_dim=1024, emb_dim=192,  # linear layers at the input for embedding projections
                 enc_dropout=0.1, encoder_kernel_size_list=[3, 7, 15, 31, 31, 31, 31], encoder_glu=True,
                 weight_softmax=True, encoder_attention_heads=8, encoder_ff_size=2048,

                 dec_input_size=192, segments=2, N=24, k=3, coprimes=(3, 5, 11, 13), cycles=(4, 6, 8, 10, 12),
                 dec_use_transformer=True, transformer_ff_size=1024, dec_activation='gelu', dec_dropout=0.1,
                 padding_idx=0x00,
                 ):
        super(DynConvColModel, self).__init__()
        self.seq_len = seq_len
        self.lang_seq_len = lang_seq_len
        self.vocab_size = vocab_size
        # needs the Embedding matrix
        # with torch.no_grad():
        self.embeds = nn.Embedding(*embed_matrix.shape)
        self.embeds.weight.data.copy_(torch.from_numpy(embed_matrix))
        # self.embeds.requires_grad = False
        self.embeds.requires_grad_(False)
        # self.embeds = self.embeds
        # Input projection of the embedding
        self.lin = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hidd_dim)),
            weight_norm(nn.Linear(hidd_dim, emb_dim)),
        )

        # TODO here the Dynamic Layers, this is an ugly hack for the fairseq inclusion in this module
        #
        self.embed_scale = math.sqrt(emb_dim)
        args = argparse.Namespace()
        args.dropout = 0.1
        args.encoder_conv_type = "dynamic"
        args.encoder_kernel_size_list = encoder_kernel_size_list
        args.encoder_glu = encoder_glu
        args.weight_softmax = weight_softmax
        args.encoder_attention_heads = encoder_attention_heads
        args.weight_dropout = enc_dropout
        args.relu_dropout = 0.
        args.input_dropout = 0.
        args.encoder_normalize_before = False
        # args.encoder_normalize_after = True
        args.encoder_layers = len(encoder_kernel_size_list)
        args.encoder_ffn_embed_dim = encoder_ff_size
        args.encoder_embed_dim = emb_dim
        args.encoder_conv_dim = args.encoder_embed_dim
        args.max_source_positions = 4096  # , 8192, 16384
        base_architecture(args)

        self.embed_positions = PositionalEmbedding(args.max_source_positions, emb_dim, padding_idx=padding_idx)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            LightConvEncoderLayer(args, kernel_size=args.encoder_kernel_size_list[i])
            for i in range(args.encoder_layers)
        ])

        # # decoder layers
        # Decoding separation in sequence length for language detection AND language model
        self.lang_lin = weight_norm(nn.Linear(seq_len, lang_seq_len))
        # self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, dim_feedforward=1024,
        #                                                activation='gelu', dropout=dec_dropout)
        # self.lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, activation='gelu')
        lang_transformer = TransformerEncoderLayer(self.lang_seq_len, 4, activation='gelu')
        self.lang_transformer = TransformerEncoder(lang_transformer, 3)

        self.lm_lin = weight_norm(nn.Linear(seq_len, seq_len))  # this is just a linear transformation
        # self.lm_transformer = TransformerEncoderLayer(seq_len, 8, 2048, activation='gelu')
        lm_transformer = TransformerEncoderLayer(seq_len, 8, 2048, activation='gelu')
        self.lm_transformer = TransformerEncoder(lm_transformer, 3)
        # Embedding projection decoding
        # TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu")
        self.decoder = UTF8SparseDecoderModule(dec_input_size, segments, N, k, coprimes, cycles, dec_use_transformer,
                                               transformer_ff_size, dec_activation, dec_dropout)
        # # needs the FAISS decoder -> TODO implement and send to GPU
        # self.indexl2 = faiss.IndexFlatL2(embed_matrix.shape[1])
        # self.indexl2.add(embed_matrix)
        self.sfmx_lin = weight_norm(nn.Linear(in_dim, vocab_size))
        # self.softmax = nn.LogSoftmax(dim=-1)

    def train(self, mode=True):
        super().train(mode)
        self.embeds.requires_grad_(False)

    def forward(self, x_in, decode_faiss=False):
        # [batch, sequence (long)]
        # Embedding
        x = self.embeds(x_in)
        # [batch, sequence, embed]

        # CHECK THIS positional embedding
        # xe = self.embed_scale * x
        # if self.embed_positions is not None:
        #     xe += self.embed_positions(x)
        # x = xe
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # Now adapt for convolutions
        x = self.lin(x)
        # work in sequence (time) dimension
        # B x T x C -> T x B x C for DynamicConvolution Fairseq module
        x = x.transpose(0, 1)
        # x = x.transpose(1, 2)
        # [sequence, batch, embed]
        x = self.layers(x, None)
        x = x.permute(1, 2, 0).continuous()
        # [batch, embed, sequence]  # needed for linear layers
        # Apply the transformer layers here
        x_lang = self.lang_lin(x)
        # x_lang = self.lang_transformer(x_lang)
        x_lm = self.lm_lin(x)
        # x_lm = self.lm_transformer(x_lm)
        # concatenate over the sequence
        x = torch.cat([x_lm, x_lang], dim=-1)

        # Go back to space dimension only (character level) for decoder
        # x = x.transpose(0, 1).contiguous()
        x = x.transpose(1, 2).contiguous()
        # [batch, sequence, embed]
        x = self.decoder(x)
        # mapping to dimension for softmax layer
        x = self.sfmx_lin(x)
        # decoding (should change for FAISS instead when I make it work correctly
        # x = self.softmax(x)
        x = F.log_softmax(x, dim=-1)
        x_lang = x[:, self.seq_len:, :].contiguous()
        x_lm = x[:, :self.seq_len, :].contiguous()
        # if decode_faiss:
        #     k = 1
        #     D, I = self.indexl2.search(x.view(-1, x.shape[-1]), k)
        #     return I.view(* x.shape[:-1])
        # print(self.seq_len, self.lang_seq_len, x.shape, x_lm.shape, x_lang.shape)
        return x_lm, x_lang
