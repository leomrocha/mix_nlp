import math
import numpy as np
import torch
from torch.nn import functional as F


###
# Taken from https://github.com/facebookresearch/XLM  (though there are not many ways of doing this)
def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
        for pos in range(n_pos)
    ])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False

###
# From my own Fibonacci based position encoding


FibArray = [0, 1]


def fib(n):
    if n < 0:
        print("Incorrect input")
    elif n <= len(FibArray):
        return FibArray[n - 1]
    else:
        temp_fib = fib(n - 1) + fib(n - 2)
        FibArray.append(temp_fib)
        return temp_fib


def get_fib_coord_emb(shape=(1024, 22), fibinit=6):
    """
    Computes #channels coordinates for a vector of the given length.
    The coordinates are computed as follow,:
        if fibinit > 0 uses shape[1] elements from fibonacci series starting from fibinit in the series
         and computes the sine & cosine for the #fibonacci values in 0->2*PI
        else computes the sine & cosine the 0->2*PI range for each value in 1->shape[1]
    @param shape: shape (length,channels) of the embedding vector
    @param fibinit: if 0 uses linear, if >0 uses fibonacci series
    @return: a vector of shape of the input value
    """

    assert (len(shape) == 2 and shape[0] > 100 and shape[1] > 0)
    ncoords = shape[1] // 2
    d_coord = shape[0]
    # get steps
    if fibinit > 0:
        # Fibonacci numbers so the signals can mix and give longer relations and have absolute like ordering which can
        # be used for longer sentences than the given input
        fib(ncoords + fibinit)
        steps = FibArray[fibinit:ncoords + fibinit]
    else:
        # Linear relations so the signals are more time independent and there is only relative ordering into the
        # input vector only
        steps = [d_coord // (i + 1) for i in range(ncoords)]
    PI2 = 2 * np.pi

    ret = []
    for stp in steps:
        arr = np.arange(0, PI2, PI2 / float(stp))
        oarr = np.tile(arr, int(np.ceil(float(d_coord) / stp)))
        ret.append(oarr[:d_coord])

    sret = torch.FloatTensor(np.stack([np.sin(ret), np.cos(ret)]))
    return sret


###############
# from HuggingFace https://github.com/huggingface/transformers BERT implementation
def gelu_old(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
# end from HuggingFace
###############

###############
# Counting number of parameters
# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
# https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############


def get_activation_fn(activation):
    if activation == "sigmoid":
        return F.sigmoid
    elif activation == "tanh":
        return F.tanh
    elif activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        return None
        # raise RuntimeError("activation should be sigmoid/tanh/relu/gelu, not %s." % activation)
