import torch
import numpy as np
import pickle
from collections import OrderedDict
"""
recommended usage:
...
from utf8_encoder import create_tables, add_mappings, save_obj

tables = create_tables(segments=2)
tables = add_mappings(tables)
recommended naming when saving separately each part:
np.save("utf8_code_matrix_2seg", tables[0])
save_obj(tables[1], "txt2code_2seg.pkl")
save_obj(tables[2], "code2txt_2seg.pkl")
save_obj(tables[3], "txt2num_2seg.pkl")
save_obj(tables[4], "num2txt_2seg.pkl")
"""

SEGMENT_INDICES_START = [0, 4, 256+4, 64+256+4, 2*64 + 256+4]
SEGMENT_INDICES_END = [4, 256+4, 64+256+4, 2*64 + 256+4, 3*64 + 256+4]


# this part makes sure to encode as bin
eye4 = np.eye(4)
eye64 = np.eye(64)
eye128 = np.eye(128)
eye256 = np.eye(256)


# TODO I need to find a more efficient way of doing this that could make this as vector or matrix operations instead
def encode_utf8_multihot(c, segments=4):
    e_c = list(c.encode())
#     code = np.zeros(36)  # 32 is the size of the input code + 4 of the extra redundancy
    nbytes = len(e_c)
    # assert( 0<nbytes && nbytes<=4)
    assert(4 >= nbytes > 0)
    bin4 = eye4[nbytes-1]  # this adds redundant knowledge about the  part
    # this is ugly but explicit, for the moment is good enough and I can see what is
#     code[:4] = bin4
    # max size of each part of the code
    # I will treat the first byte as always 8 bits, this will make it easier to decode later and adds aditional information
    # this has an extra benefit, when a code is there only certain regions will become 1 giving an extra hint to the network
    # maxsizes = [2**8, 2**6, 2**6, 2**6]
    code = np.zeros(4 + (2 ** 8) + (segments - 1) * (2 ** 6))
    # code = np.zeros(4 + (2 ** 7) + (segments - 1) * (2 ** 6))
    masks = [0xff, 0x3f, 0x3f, 0x3f]
    indices = [256+4, 64+256+4, 2*64 + 256+4, 3*64 + 256+4]  # end indices of each segment
    # indices = [128 + 4, 64 + 128 + 4, 2 * 64 + 128 + 4, 3 * 64 + 128 + 4]  # end indices of each segment
    maxsizes = [eye256, eye64, eye64, eye64]
    # maxsizes = [eye128, eye64, eye64, eye64]
    masks = masks[:segments]
    indices = indices[:segments]
    maxsizes = maxsizes[:segments]
    # print(len(masks), len(indices), masks, indices)

    code[:4] = bin4
    prev_i = 4
    for i, n, e, m in zip(indices[:nbytes], e_c, maxsizes[:nbytes], masks[:nbytes]):
        code[prev_i:i] = e[n & m]  # masking
        prev_i = i
    return code


# masks make values for utf-8 valid, the process is first adding the missing bits for the valid encoding,
# and then subtracting the ones that should not be there
and_mask1 = [0b01111111, 0x00, 0x00, 0x00]  # and mask
# or_mask1 = [0x00, 0x00, 0x00, 0x00]
and_mask2 = [0b11011111, 0b10111111, 0x00, 0x00]
or_mask2 = [0b11000000, 0b10000000, 0x00, 0x00]

and_mask3 = [0b11101111, 0b10111111, 0b10111111, 0x00]
or_mask3 = [0b11100000, 0b10000000, 0b10000000, 0x00]

and_mask4 = [0b11110111, 0b10111111, 0b10111111, 0b10111111]
or_mask4 = [0b11110000, 0b10000000, 0b10000000, 0b10000000]


def create_tables(segments=4):
    assert(0 < segments <= 4)
    # will create a table with all the codings -> this one
    # and dictionaries with the mappings
    code_matrix = []
    code_count = 0
    except_count = 0
    txt2code = OrderedDict()  # keeps a mapping from txt character to the code
    code2txt = OrderedDict()  # keeps a mapping from  the code to the original txt character
    txt2num = OrderedDict()  # from character to a coded index number for the table (for use in torch.nn.F.Embedding?)
    num2txt = OrderedDict()  # keeps a mapping from  the index in the table to the original character
    # to encode we need to take in account that there are 4 segments

    def append_code(txt, index):
        multihot = encode_utf8_multihot(txt, segments)
        code_matrix.append(multihot)
        txt2code[txt] = multihot
        code2txt[bytes(multihot)] = txt
        txt2num[txt] = index
        num2txt[index] = txt

    # max number of elements that can be created in the bytes 2 3 and 4 of utf-8 codes
    max_6b = 2 ** 6
    # mask1 = 0b01111111
    # max1 = 0xff & mask1
    # generate all values for the first segment,
    for i in range(2**7):  # is the same as max1
        txt = str(bytes([i]), 'utf-8')
        # code_count
        append_code(txt, code_count)
        code_count += 1

    if segments >= 2:
        # generate all values for the second segment,
        # index_offset_2 = 128
        max2_a = 0xff & 0b00011111
        for i in range(max2_a):
            for j in range(max_6b):
                # index =
                try:
                    txt = str(bytes([i | or_mask2[0], j | or_mask2[1]]), 'utf-8')
                    # index = index_offset_2 + i*j
                    append_code(txt, code_count)
                    code_count += 1
                except Exception as e:
                    # print(i, j, i | or_mask2[0], j | or_mask2[1])
                    except_count +=1
                    # raise e
                    pass
    if segments >= 3:
        # generate all values for the third segment,
        # index_offset_3 = index_offset_2 + (max2_a * max_6b)
        max3_a = 0xff & 0b00001111
        for i in range(max3_a):
            for j in range(max_6b):
                for k in range(max_6b):
                    try:
                        txt = str(bytes([i | or_mask3[0], j | or_mask3[1], k | or_mask3[2]]), 'utf-8')
                        # index = index_offset_3 + i*j*k
                        append_code(txt, code_count)
                        code_count += 1
                    except Exception as e:
                        # print(i, j, i | or_mask2[0], j | or_mask2[1])
                        except_count += 1
                        # raise e
                        pass

    if segments == 4:
        # generate all values for the fourth segment,
        # index_offset_4 = index_offset_3 + (max3_a * max_6b)
        max4_a = 0xff & 0b00000111
        for i in range(max4_a):
            for j in range(max_6b):
                for k in range(max_6b):
                    for l in range(max_6b):
                        try:
                            txt = str(bytes([i | or_mask4[0], j | or_mask4[1], k | or_mask4[2], l | or_mask4[3]]), 'utf-8')
                            # index = index_offset_4 + i*j*k*l
                            append_code(txt, code_count)
                            code_count += 1
                        except Exception as e:
                            # print(i, j, i | or_mask2[0], j | or_mask2[1])
                            except_count += 1
                            # raise e
                            pass
    print("number of codes = ", code_count)
    print("number of code_exceptions = ", except_count)
    code_matrix = np.stack(code_matrix)  # the issue here is that is a sparse matrix but I'm working as if  dense ...

    return code_matrix, txt2code, code2txt, txt2num, num2txt


def add_mappings(codebook, mappings={"<start>": 0x02, "<end>": 0x03, "<unk>": 0x15}):
    """
    :param codebook: input codebook to which to add special codes, it consists of a 5-tuple
    (code_matrix, txt2code, code2txt, txt2num, num2txt)
    :param mappings: text to map and the index in the code_matrix that is referring too
    :return: a 5-tuple (code_matrix, txt2code, code2txt, txt2num, num2txt) with the last elements set as codelist
    and an extra dimension to signal the special codes
    """
    assert len(codebook) == 5
    code_matrix, txt2code, code2txt, txt2num, num2txt = codebook
    # modify only the dictionaries mapping the text:
    for k, v in mappings.items():
        txt2code[k] = code_matrix[v]
        code2txt[bytes(code_matrix[v])] = k
        txt2num[k] = v
        num2txt[v] = k
    return code_matrix, txt2code, code2txt, txt2num, num2txt


# from https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809
def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

