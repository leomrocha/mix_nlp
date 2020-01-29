import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils import weight_norm
try:
    from .tools import get_activation_fn
except:
    # to solve issue with ipython executing this import
    from tools import get_activation_fn


class Conv1DBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, nlayers=3, dropout=0.1, groups=8, activation="relu"):
        """
        :param c_in: input channels
        :param c_out: output channels
        :param kernel_size:
        :param nlayers: number of convolutional layers per block
        :param dropout:
        :param groups: number of groups as in filter groups
        :param activation: activation function to use at the end of the convolutional block
        """
        super(Conv1DBlock, self).__init__()

        if c_in == c_out:
            self.use_proj = False
        else:
            self.use_proj = True

        self.convresid = weight_norm(nn.Conv1d(c_in, c_out, 1))  # [down|up]sample for residual connection if needed

        self.convs = []
        for i in range(nlayers):
            t_c_in = c_out
            if i == 0:
                t_c_in = c_in
            # Left padding
            # pad = nn.ConstantPad1d((kernel_size - 1) // 2, 0)
            cnv = weight_norm(nn.Conv1d(t_c_in, c_out, kernel_size, padding=(kernel_size - 1) // 2, groups=groups))
            # cnv = weight_norm(nn.Conv1d(t_c_in, c_out, kernel_size, groups=groups))
            # self.convs.append(pad)
            self.convs.append(cnv)

        self.convs = nn.Sequential(*self.convs)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, x):
        # not use padding -> is up to the main network to decide
        # res = self.leftpad(x)
        # print("1 conv1b", x.shape, x.dtype, x.is_cuda)
        res = x
        # residual connection channel dimension adaptation
        if self.use_proj:  # if in_c != out_c, need to change size of residual
            res = self.convresid(res)
        # print("2 conv1b", res.shape)
        out = self.convs(x)
        # print("3 conv1b", out.shape)
        out = self.dropout(out)
        # print("4 conv1b", out.shape)
        out = self.activation(out + res)
        # out = F.layer_norm(out, out.shape)
        return out


class ConvColumn(nn.Module):
    """

    """
    def __init__(self, c_in=[192, 256, 512, 1024, 512, 192], c_out=[192, 256, 512, 1024, 512, 192],  # channels for blocks
                 b_layers=[3, 5, 5, 5, 3],  # number of layers for each bloc
                 first_k_size=3, kernel_size=3,
                 dropout=0.3, groups=4,
                 activation="gelu",  # "sigmoid",
                 ):
        """
        Convolutional column composed of multiple sequential convolutional blocks
        :param c_in:
        :param c_out:
        :param b_layers:
        :param first_k_size:
        :param kernel_size:
        :param dropout:
        :param groups:
        :param activation:
        """
        super(ConvColumn, self).__init__()

        assert c_in[1:] == c_out[:-1]
        # assert len(c_out) == res_layers
        # input convolution layer is the one who adapts the input for the columns that follow
        self.conv0 = nn.Conv1d(c_in[0], c_out[0], first_k_size, padding=(kernel_size-1)//2, stride=(first_k_size-1)//2)
        self.drop0 = nn.Dropout(dropout)

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        # self.maxpool_blocks = nn.ModuleList()
        for cin, cout, lays in zip(c_in[1:], c_out[1:], b_layers):
            cnv = Conv1DBlock(cin, cout, kernel_size, lays, dropout, groups)
            # mp = nn.MaxPool1d(2, stride=2)
            self.conv_blocks.append(cnv)
            # self.maxpool_blocks.append(mp)
        self.convolutions = nn.Sequential(*self.conv_blocks)
        #
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)  # nn.Sigmoid()  # Better for multihot than relu

    def forward(self, x):
        ret = self.conv0(x)
        ret = self.drop0(ret)
        ret = self.convolutions(ret)
        ret = self.dropout(ret)
        ret = self.activation(ret)
        return ret


class MixtureEncoderBlock(nn.Module):
    """
    Block that uses convolutions, linear layers and transformers to work on time and embedding space iteratively
    to keep computation and number of channels
    """
    def __init__(self, in_dim=2048, out_dim=1024,  # sequence (temporal) dimension
                 lin_in=192, lin_hidd=1024, lin_out=192,  # linear layers
                 c_in=192, c_out=1024, conv_nlayers=5, conv_groups=8, conv_dropout=0.2,  # convolutional block
                 conv_kern_size=3, conv_activation="gelu",
                 downsample_stride=2, chann_downsample_stride=2,
                 transf_dim=128, transf_nheads=8, transf_dim_feedforward=1024,  # Transformer channel-wise
                 transf_dropout=0.1, transf_activation="gelu",  # Transformer
                 # o_lin_in=512,
                 o_lin_hidd=1024,
                 # o_lin_out=192  # output projection for final dimensionality reduction
                 ):
        """"""
        super(MixtureEncoderBlock, self).__init__()
        # "highway" projection for gradient facilitation ... although this might better be a couple of 1x1 conv layers
        # self.skip_projection_time = weight_norm(nn.Linear(in_dim, out_dim))
        # self.skip_projection_embed = weight_norm(nn.Linear(lin_in, o_lin_out))

        # channel-wise linear layers for input projections
        self.linear_in = nn.Sequential(
            weight_norm(nn.Linear(lin_in, lin_hidd)),
            weight_norm(nn.Linear(lin_hidd, lin_out)),
        )
        self.lin_norm1 = nn.LayerNorm(lin_out)
        # self.lin_norm1 = nn.LayerNorm(in_dim)
        # Convolution Block temporal (sequence) dimension
        self.conv = Conv1DBlock(c_in, c_out, kernel_size=conv_kern_size, nlayers=conv_nlayers,
                                dropout=conv_dropout, groups=conv_groups, activation=conv_activation)
        self.conv_norm1 = nn.LayerNorm(in_dim)
        # MaxPool - Downsample temporal (sequence) dimension with strided convolution
        self.max_pool_seq = nn.MaxPool1d(stride=downsample_stride, kernel_size=downsample_stride)

        # Channel-Wise Transformer Layer for dimension reduction,
        assert c_out % transf_dim == 0  # check that we'll fit the transformer on exact chunks
        self.transf_dim = transf_dim
        self.chann_transformer = TransformerEncoderLayer(d_model=transf_dim, nhead=transf_nheads,
                                                         dim_feedforward=transf_dim_feedforward, dropout=transf_dropout,
                                                         activation=transf_activation)
        assert in_dim % downsample_stride == 0
        self.transf_norm1 = nn.LayerNorm(in_dim // downsample_stride)
        # channel-wise max-pool for dimensionality reduction
        self.max_pool_chann = nn.MaxPool1d(stride=chann_downsample_stride, kernel_size=chann_downsample_stride)
        # Final downsampling projection
        assert c_out % chann_downsample_stride == 0
        o_lin_in = c_out // chann_downsample_stride
        o_lin_out = lin_in
        self.linear_out = nn.Sequential(
            weight_norm(nn.Linear(o_lin_in, o_lin_hidd)),
            weight_norm(nn.Linear(o_lin_hidd, o_lin_out)),
        )
        self.lin_norm2 = nn.LayerNorm(o_lin_out)
        # time-wise transformer "convolution" TODO
        self.out_norm = nn.LayerNorm(o_lin_out)

    def forward(self, x):
        ##
        # compute the projections for "skip" connection (makes it easier for gradient to flow)
        # [batch, sequence, embed] -> [batch, 2048, 192]
        # y = self.res_projection_embed()
        # y = self.res_projection_time(y.transpose(1, 2))
        # # y [batch, embed, sequence] -> [batch, 192, 1024]
        # y = y.transpose(1, 2)
        # y [batch, sequence, embed] -> [batch, 1024, 192]
        ##
        # [batch, sequence, embed] -> [batch, 2048, 192]
        # project input such as to have a better representation channel-wise
        y = x  # linear residual
        # print(x.shape)
        x = self.linear_in(x)
        print(x.shape, y.shape)
        x = self.lin_norm1(x + y)
        x = x.transpose(1, 2)
        # [batch, embed, sequence]  -> [batch, 192, 2048]
        # convolutional block, many parameters here
        x = self.conv(x)
        x = self.conv_norm1(x)
        # [batch, embed, sequence]  -> [batch, 1024, 2048]
        # Reduce temporal dimension
        x = self.max_pool_seq(x)
        # [batch, embed, sequence]  -> [batch, 1024, 1024]
        # Prepare for reducing channel dimensions
        x = x.transpose(1, 2)
        # [batch, sequence, embed] -> [batch, 1024, 1024]
        # Attention per chunk, apply the same transformer (share parameters) for each part.
        chunks = []
        for c in torch.split(x, self.transf_dim, dim=-1):
            chunks.append(self.chann_transformer(c))
        x = torch.cat(chunks, dim=-1)
        x = self.transf_norm1(x)
        # [batch, sequence, embed] -> [batch, 1024, 1024]
        x = self.max_pool_chann(x)
        # [batch, sequence, embed] -> [batch, 1024, 512]
        # y = x  # linear residual
        x = self.linear_out(x)
        x = self.lin_norm2(x)
        # x = self.lin_norm2(x + y)
        # [batch, sequence, embed] -> [batch, 1024, 192]
        # x = self.out_norm(x + y)
        return x
