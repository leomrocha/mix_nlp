"""
This file contains the functions that execute every aspect of the paper,
each function contains the chosen configuration values and is done like that mostly for reproducibility WITHOUT the need
of an external configuration file, so it is purposely written with hardcoded values instead of depending on an external
file.
Many things might (and will be) done in a non-optimal code or production-level code in this file for the sake of clarity
and/or idea separation.
Code is single threaded single process, this is to measure build and run times and compare results (also parallel code
can be slightly or much more complex to run and debug)
"""
from itertools import combinations
import os
import pickle
import numpy as np
import scipy as sp
import scipy.sparse
import time
import unicodedata
import unidecode

import torch.nn.functional as F

try:
    from .utf8_encoder import *
    from .constants import *
except:
    # to solve issue with ipython executing this import
    from utf8_encoder import *
    from constants import *

SEGMENTS = [1, 2, 3, 4]
NCODES = [128, 1984, 59328, 1107904]
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]


def create_measure_tables():
    elapsed_times = []
    tables = []
    shapes = []
    # exceptions = []
    sizes = []
    matrix_sizes = []
    sparse_sizes = []
    paths = []
    base_name = "codes/utf8_codes-{}seg.pkl"
    matrix_name = "codes/utf8_codes_matrix-{}seg.npy"
    sparse_matrix_name = "codes/utf8_codes_sparse_matrix-{}seg.npy"
    for i in range(1, 5):
        t_init = time.time()
        t = create_tables(segments=i)
        t = add_mappings(t)
        tables.append(t)
        # Save the shape of the matrices
        shapes.append(t[0].shape)
        # Save tables
        name = base_name.format(i)
        paths.append(name)
        with open(name, 'wb') as f:
            pickle.dump(t, f, pickle.HIGHEST_PROTOCOL)
        # Save matrix alone
        mname = matrix_name.format(i)
        np.save(mname, t[0])
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(i)
        spcodes = sp.sparse.coo_matrix(t[0])
        np.save(smname, spcodes)
        # Measure size in disk in MB
        mb = os.path.getsize(name) / (1024 ** 2)
        sizes.append(mb)
        # Dense Matrix
        mmb = os.path.getsize(mname) / (1024 ** 2)
        matrix_sizes.append(mmb)
        # Sparse Matrix
        smmb = os.path.getsize(smname) / (1024 ** 2)
        sparse_sizes.append(smmb)
        t_end = time.time()
        el = t_end - t_init
        elapsed_times.append(el)

    col = "| Segments | exec_time (sec) |  matrix_shape | Size in Disk (MB): | Matrix Size in Disk (MB): \
           | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {:.3f} | {} | {:.2f} | {:.2f} | {:.2f} | {} |"

    print(col)
    for i, et, sh, si, ms, ss, p in zip(range(1, 5), elapsed_times, shapes, sizes, matrix_sizes, sparse_sizes, paths):
        print(row.format(i, et, sh, si, ms, ss, p))


def overfit_tests():
    pass


def sparse_Nk_dimension_analysis():
    # find the minimum N and k for which the condition is filled for the different codes
    results = []
    for code_points in NCODES:
        for k in range(2, 7):
            for N in range(10, 256):
                v = int(np.prod(list(range(N, N - k, -1))) / np.prod(list(range(1, k + 1))))
                if v > code_points:
                    # print("code_size={}; N={},k={}".format(v, N, k))
                    results.append((code_points, v, N, k, '{:.3f}'.format(k / N)))
                    break
    return results


def sparse_code_Nk(code_size, N, k):
    # get the indices for the ones
    comb = combinations(list(range(N)), k)  # iterator
    # limit to code size
    comb = np.array(list(comb))[:code_size]
    # compute referential as flat index to be able to use for
    comb = comb + np.array(range(code_size))[:comb.shape[0]].reshape([-1, 1]) * N
    # convert indices to dense binary matrix
    sc = np.zeros([code_size, N], dtype=bool)
    np.put(sc, comb, 1)
    return sc


def create_sparse_Nk_codes():
    col = "| Segments | code size | Vector Size | N | k |exec_time (sec) |  Matrix Size in Disk (MB): \
               | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_sparse_codes-{}_N-{}_k-{}_seg"
    sparse_matrix_name = "codes/utf8_sparse_codes-{}_N-{}_k-{}_seg_sparse-matrix"
    params = [(NCODES[0], 17, 2), (NCODES[1], 24, 3), (NCODES[2], 37, 4), (NCODES[3], 45, 5)]
    codes = []
    print(col)
    for i, p in enumerate(params):
        t_init = time.time()
        nc, N, k = p
        scode = sparse_code_Nk(nc, N, k)
        codes.append(scode)
        # Save matrix alone
        mname = base_name.format(i + 1, N, k)
        np.save(mname, scode)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(i + 1, N, k)
        spcodes = sp.sparse.coo_matrix(scode)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init
        print(row.format(i + 1, nc, scode.shape, N, k, el, msize, smsize, mname))
    return codes


def multihot_primes_conf_finder():
    all_codes = []
    for i in range(2, 5):
        arr = list(combinations(PRIMES, i))
        for a in arr:
            ll = len(a)
            ncodes = np.prod(a)
            vsize = np.sum(a)
            sparsity = round(i / vsize, 3)
            all_codes.append((a, ll, sparsity, vsize, ncodes))
    all_codes = sorted(all_codes, key=lambda x: x[-2])
    codes_1seg = [i for i in all_codes if i[-1] > NCODES[0] and i[-1] < NCODES[3] * 2]
    codes_2seg = [i for i in all_codes if i[-1] > NCODES[1] and i[-1] < NCODES[3] * 2]
    codes_3seg = [i for i in all_codes if i[-1] > NCODES[2] and i[-1] < NCODES[3] * 2]
    codes_4seg = [i for i in all_codes if i[-1] > NCODES[3] and i[-1] < NCODES[3] * 2]
    return all_codes, codes_1seg, codes_2seg, codes_3seg, codes_4seg


def generate_multihot_prime_code(ncodes, subcode_list):
    eyes = [np.eye(c).astype(bool) for c in subcode_list]
    # TODO
    # stack each and cut to the ncodes size
    cols = [np.tile(e, ((ncodes // e.shape[0]) + 1, 1))[:ncodes] for e in eyes]
    code = np.hstack(cols)
    return code


def all_multihot_primes():
    col = "| Segments | code size | Vector Size | primes |exec_time (sec) |  Matrix Size in Disk (MB): \
                   | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_coprime_codes-{}_primes-{}_{}_seg"
    sparse_matrix_name = "codes/utf8_coprime_codes-{}_primes-{}_{}_seg_sparse-matrix"
    # the codes already were selected by hand from the ones and are:
    code_config = [(NCODES[0], (3, 5, 11)),
                   (NCODES[1], (3, 5, 11, 13)),
                   (NCODES[2], (11, 13, 19, 23)),
                   (NCODES[3], (23, 31, 37, 43))]
    codes = []
    print(row)
    for i, cc in enumerate(code_config):
        t_init = time.time()
        code = generate_multihot_prime_code(*cc)
        codes.append(code)
        # Save matrix alone
        mname = base_name.format(cc[0], str(cc[1]), i + 1)
        np.save(mname, code)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(cc[0], str(cc[1]), i + 1)
        spcodes = sp.sparse.coo_matrix(code)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init

        print(row.format(i + 1, cc[0], code.shape, cc[1], el, msize, smsize, mname))
    return codes


def create_choose_Nk_coprimes_codes():
    col = "| Segments | code size | Vector Size | N | k | primes |exec_time (sec) |  Matrix Size in Disk (MB): \
                       | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_N-{}k-{}-coprime_codes-{}_primes-{}_{}_seg"
    sparse_matrix_name = "codes/utf8_N-{}k-{}-coprime_codes-{}_primes-{}_{}_seg_sparse-matrix"
    # N choose k + coprime multihot
    # code dim, N,k,target dim, prime dim, primes
    config = [(NCODES[0], 17, 2, 32, 15, (3, 5, 7)),
              (NCODES[1], 24, 3, 48, 24, (5, 8, 11)),
              (NCODES[2], 37, 4, 64, 27, (3, 5, 8, 11)),
              (NCODES[3], 45, 5, 96, 51, (3, 7, 11, 13, 17))]
    codes = []
    print(col)
    for i, (cs, N, k, tgts, ms, coprimes) in enumerate(config):
        t_init = time.time()
        nk = sparse_code_Nk(cs, N, k)
        cp = generate_multihot_prime_code(cs, coprimes)
        code = np.hstack([nk, cp])
        codes.append(code)
        # Save matrix alone
        mname = base_name.format(N, k, cs, str(coprimes), i + 1)
        np.save(mname, code)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(N, k, cs, str(coprimes), i + 1)
        spcodes = sp.sparse.coo_matrix(code)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init

        print(row.format(i + 1, cs, code.shape, N, k, coprimes, el, msize, smsize, mname))

    return codes


def create_coprimes_choose_Nk_codes():
    col = "| Segments | code size | Vector Size | N | k | primes |exec_time (sec) |  Matrix Size in Disk (MB): \
                       | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_coprime_codes-{}_primes-{}_N-{}k-{}_{}-seg"
    sparse_matrix_name = "codes/utf8_coprime_codes-{}_primes-{}_N-{}k-{}_{}-seg_sparse-matrix"
    # coprime multihot + N choose k
    # code dim, primes, (N, k)
    config = [(NCODES[0], (3, 5, 11), (13, 3)),
              (NCODES[1], (3, 5, 11, 13), (32, 3)),
              (NCODES[2], (11, 13, 19, 23), (30, 4)),
              (NCODES[3], (23, 31, 37, 43), (58, 5))]
    codes = []
    print(col)
    for i, (cs, coprimes, (N, k)) in enumerate(config):
        t_init = time.time()
        nk = sparse_code_Nk(cs, N, k)
        nk = np.tile(nk, ((cs // nk.shape[0]) + 1, 1))[:cs]
        cp = generate_multihot_prime_code(cs, coprimes)
        code = np.hstack([cp, nk])
        codes.append(code)
        # Save matrix alone
        mname = base_name.format(cs, str(coprimes), N, k, i + 1)
        np.save(mname, code)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(cs, str(coprimes), N, k, i + 1)
        spcodes = sp.sparse.coo_matrix(code)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init

        print(row.format(i + 1, cs, code.shape, N, k, coprimes, el, msize, smsize, mname))

    return codes


def create_specific_redundant_codes():
    col = "| Segments | code size | Vector Size | N | k | primes |exec_time (sec) |  Matrix Size in Disk (MB): \
                       | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_coprime_codes-{}_primes-{}_N-{}k-{}_{}-seg"
    sparse_matrix_name = "codes/utf8_coprime_codes-{}_primes-{}_N-{}k-{}_{}-seg_sparse-matrix"
    # coprime multihot + N choose k
    # code dim, primes, (N, k)
    config = [(NCODES[0], (3, 5, 11), (13, 3)),
              (NCODES[1], (3, 5, 11, 13), (32, 3)),
              (NCODES[2], (11, 13, 19, 23), (30, 4)),
              (NCODES[3], (23, 31, 37, 43), (58, 5))]
    codes = []
    print(col)
    for i, (cs, coprimes, (N, k)) in enumerate(config):
        t_init = time.time()
        nk = sparse_code_Nk(cs, N, k)
        nk = np.tile(nk, ((cs // nk.shape[0]) + 1, 1))[:cs]
        cp = generate_multihot_prime_code(cs, coprimes)
        code = np.hstack([cp, nk])
        codes.append(code)
        # Save matrix alone
        mname = base_name.format(cs, str(coprimes), N, k, i + 1)
        np.save(mname, code)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(cs, str(coprimes), N, k, i + 1)
        spcodes = sp.sparse.coo_matrix(code)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init

        print(row.format(i + 1, cs, code.shape, N, k, coprimes, el, msize, smsize, mname))

    return codes


# FIXME
# The cycle code generator is WRONG and MUST be corrected, I'll just not use it for the moment and that's it.
# WARNING this is NOT working, DO NOT USE
def create_single_cycle_code(code_size, sizes):
    """
    :param code_size: number of elements in the code
    :param sizes: iterable of vector sizes to include in the code
    :return: a multi-hot (or one-hot) vector of shape=(code_size, sum(sizes)) with the codes
    """
    codes = []
    for s in sizes:
        # compute index of the value '1'
        idx = np.arange(1, code_size + 1)
        idx = idx // s
        # idx = idx % s
        # convert indices to dense binary matrix
        sc = np.zeros([code_size, s], dtype=bool)
        np.put(sc, idx, 1)
        codes.append(sc)
    ret = np.hstack(codes)
    return ret


def create_codematrix_from_conf(config=[]):
    """
    The codes are N choose k + coprime + filling with single cycle method giving redundancy and 2
    complete representations
    There are only done for 2 and 3 segments, the cycles are arbitrarilly chosen to fill the gaps to the next interesting
    dimension (64/128

    :return:
    """
    col = "| Segments | code size | Vector Size | N | k | primes | cycles | exec_time (sec) |  Matrix Size in Disk (MB): \
                           | Sparse Matrix Size in Disk (MB): |code path"
    row = "| {} | {} | {} | {} | {} | {} | {} | {:.3f} | {:.2f} | {:.2f} | {} |"
    base_name = "codes/utf8_{}-seg_{}-codepoints_{}-dim_N-{}-k{}_coprimes-{}_cycles-{}_dense"
    sparse_matrix_name = "codes/utf8_{}-seg_{}-codepoints_{}-dim_N-{}-k{}_coprimes-{}_cycles-{}_sparse"
    if len(config) <= 0:
        config = [
            # segment, number of code-points, (n,k), (coprimes), (cycles), dimension, sparcity
            (2, NCODES[1], (24, 3), (3, 5, 11, 13), (6, 2), 64, 9 / 64),
            # (2, 1916, (24, 3), (3, 5, 11, 13), (6, 2), 64, 9 / 64),
            (3, NCODES[2], (37, 4), (11, 13, 19, 23), (11, 7, 4, 3), 128, 12 / 128),
        ]
    codes = []
    print(col)
    for seg, codepoints, (N, k), coprimes, cycles, dim, sparcity in config:
        t_init = time.time()
        nk = sparse_code_Nk(codepoints, N, k)
        nk = np.tile(nk, ((codepoints // nk.shape[0]) + 1, 1))[:codepoints]
        cp = generate_multihot_prime_code(codepoints, coprimes)
        cc = create_single_cycle_code(codepoints, cycles)
        code = np.hstack([cp, nk, cc])
        codes.append(code)
        # Save matrix alone
        mname = base_name.format(seg, codepoints, dim, N, k, str(coprimes), str(cycles))
        np.save(mname, code)
        # Save Sparse matrix alone
        smname = sparse_matrix_name.format(seg, codepoints, dim, N, k, str(coprimes), str(cycles))
        spcodes = sp.sparse.coo_matrix(code)
        np.save(smname, spcodes)
        msize = os.path.getsize(mname + ".npy") / (1024 ** 2)
        # Sparse Matrix
        smsize = os.path.getsize(smname + ".npy") / (1024 ** 2)
        t_end = time.time()
        el = t_end - t_init

        print(row.format(seg, codepoints, code.shape, N, k, coprimes, cycles, el, msize, smsize, mname))

    return codes


# BAD BAD these configurations go together ... need to do an automation system to compute them
# CHARSET_PATH = "codes/all_chars.chars"
# CONFIG = (2, 1916, (24, 3), (3, 5, 11, 13), (6, 2), 64, 9 / 64)
# OFNAME = "codes/adhoc-codebook-1916.pkl"
CHARSET_PATH = "codes/all_chars.chars"
CONFIG = (2, 1871, (24, 3), (3, 5, 11, 13), (4, 6, 8, 10, 12), 96, 13 / 96)
OFNAME = "codes/adhoc-codebook-1871.pkl"


def create_codebook(charset, config=CONFIG,
                    ofname=OFNAME,
                    special_codes=SPECIAL_CODES,
                    nul_row_is_zero=True,
                    reserved_spaces=RESERVED_CODE_SPACE
                    ):
    """
    :param charset_fpath: file path where the set of characters is available
    :param config: list of tuples: (segment, number of code-points, (n,k), (coprimes), (cycles), dimension, sparcity)
    :param ofname: Where to save the codebook
    :param special_codes: special codes mapping for the output dictionary
    :param nul_row_is_zero: if the first row (the NUL one) should be zeros or the given code
    :param reserved_spaces: the reserved spaces at the beginning of the codebook, 32 is the default as is the number of
    control codes in utf-8. This later is used for remapping reserved SPECIAL_CODES, IS 32
    :return:
    """
    # TODO this code is ugly but works wiht the right configuration, for the moment
    # TODO make the configuration selection automatic from some config points and the charset
    codes = create_codematrix_from_conf([config])[0]
    if nul_row_is_zero:
        # assume nul row is the first one
        codes[0, :] = 0
    # create dict
    char2int = OrderedDict()
    int2char = OrderedDict()
    # add the number of reserved chars at the beginning
    for i in range(reserved_spaces):  # Warning, must be <128
        # use utf-8 codepoints
        c = str(bytes([i]), 'utf-8')
        char2int[c] = i
        # for the reverse mapping, to avoid issues on decoding, leave them unassigned UNASSIGNED='◁???▷'
        # could use UNK but I'd rather have it be obviously different, leaving unassigned is an issue
        int2char[i] = c  # UNASSIGNED
    # overwrite the indices of the reverse mapping for the special codes
    for c, i, c_alt in special_codes:
        # Take into account this will duplicate the char2int mapping having 2 chars and the alternative code
        # mapping to the same int
        char2int[c] = i
        # char2int[c_alt] = i
        # but the int reverse index will be overwritten
        int2char[i] = c

    for i, c in enumerate(list(charset)):
        # forward the index
        j = i + reserved_spaces
        char2int[c] = j
        int2char[j] = c

    # pickle all together
    codebook = (codes, char2int, int2char)
    with open(ofname, 'wb') as f:
        print("saving file {} with codes.shape {} | char2int {} | int2char {}".format(
            ofname, codes.shape, len(char2int), len(int2char)))
        pickle.dump(codebook, f, pickle.HIGHEST_PROTOCOL)

    return codebook


##############################################################
# Last codes created that handle  compositional codes.
# characters are


def create_base_codebook(charset, special_codes=SPECIAL_CODES, code_size=2145 + 33,
                         N=24, k=3,
                         subcode_list=(2, 5, 9, 11, 13),  # subcode_list=(2,3,5,11,13),
                         # cycle_list=(2, 3),  # (4,6,10,12),  # WARNING< DO NOT USE < bug in the cycle code generator
                         nul_row_is_zero=True, reserved_spaces=RESERVED_CODE_SPACE
                         ):
    """
    :param charset:
    :param special_codes:
    :param code_size:
    :param N:
    :param k:
    :param subcode_list: configuration for the prime configuration
    :param special_codes: special codes mapping for the output dictionary
    :param nul_row_is_zero: if the first row (the NUL one) should be zeros or the given code
    :param reserved_spaces: the reserved spaces at the beginning of the codebook, 32 is the default as is the number of
    control codes in utf-8. This later is used for remapping reserved SPECIAL_CODES, IS 32
    :return:
    """
    # TODO this code is ugly but works with the right configuration, for the moment
    # TODO make the configuration selection automatic from some config points and the charset
    codes = [
        sparse_code_Nk(code_size, N, k),
        generate_multihot_prime_code(code_size, subcode_list),
        #         create_single_cycle_code(code_size, cycle_list),  # this code generator is only for redundancy

    ]
    if nul_row_is_zero:
        # assume nul row is the first one
        for code in codes:
            code[0, :] = 0
    # create dict
    char2int = OrderedDict()
    int2char = OrderedDict()
    # add the number of reserved chars at the beginning
    for i in range(reserved_spaces):  # Warning, must be <128
        # use utf-8 codepoints
        c = str(bytes([i]), 'utf-8')
        char2int[c] = i
        # for the reverse mapping, to avoid issues on decoding, leave them unassigned UNASSIGNED='◁???▷'
        # could use UNK but I'd rather have it be obviously different, leaving unassigned is an issue
        int2char[i] = c  # UNASSIGNED
    # overwrite the indices of the reverse mapping for the special codes
    for c, i, c_alt in special_codes:
        # Take into account this will duplicate the char2int mapping having 2 chars and the alternative code
        # mapping to the same int
        char2int[c] = i
        # char2int[c_alt] = i
        # but the int reverse index will be overwritten
        int2char[i] = c

    for i, c in enumerate(list(set(charset))):
        # forward the index
        j = i + reserved_spaces
        char2int[c] = j
        int2char[j] = c

    # pickle all together
    codebook = (codes, char2int, int2char)
    #     with open(ofname, 'wb') as f:
    #         print("saving file {} with codes.shape {} | char2int {} | int2char {}".format(
    #             ofname, codes.shape, len(char2int), len(int2char)))
    #         pickle.dump(codebook, f, pickle.HIGHEST_PROTOCOL)
    return codebook


# from
# https://stackoverflow.com/questions/517923/what-is-the-best-way-to-remove-accents-in-a-python-unicode-string

def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def get_code_item(c, codebook, padded_codebook, circ_padded_codebook, char2int):
    """
    Converts a char sequence to a code dictionary for compositional code generation

    The idea behind the scenes is to generate several codes from convolutional and sum of the characters.
    The final code is decided by a post-processing step.

    The codebook for the input should be able to encode ALL values given as input, in this idea there is no exception
    handling and an exception is expected if a symbol is not present.

    :param c:
    :param codebook:
    :param padded_codebook:
    :param circ_padded_codebook:
    :param char2int:
    :return:
    """
    # convert to lowercase for the symbol representation
    #
    c_len = len(c)
    c = unicodedata.normalize('NFKD', c)
    nc = c.lower()
    ac = remove_accents(nc)

    nc_vecs = [codebook[char2int[i]] for i in nc]
    ac_vecs = [codebook[char2int[i]] for i in ac]

    # padded version to be able to convolve later
    nc_padded = [padded_codebook[char2int[i]] for i in nc]
    ac_padded = [padded_codebook[char2int[i]] for i in ac]

    nc_cpadded = [circ_padded_codebook[char2int[i]] for i in nc]
    ac_cpadded = [circ_padded_codebook[char2int[i]] for i in ac]

    # circular convolution -> keeps order of elements in token
    nc_conv = nc_padded[0] if len(nc_padded) > 0 else codebook[0]
    if len(nc_padded) > 1:
        #         print(nc_conv.shape)
        for cpadded in nc_cpadded[1:]:
            #             print(c, padded.shape, nc_conv.shape)
            olen = nc_conv.shape[0]  # original vector length before
            nc_conv = np.convolve(nc_conv, cpadded, mode='same')
            #
            nc_conv = nc_conv if nc_conv.shape[0] == olen else nc_conv[olen // 2:-olen // 2]

    ac_conv = ac_padded[0] if len(ac_padded) > 0 else codebook[0]
    if len(ac_conv) > 1:
        for cpadded in ac_cpadded[1:]:
            olen = ac_conv.shape[0]  # original vector length before
            ac_conv = np.convolve(ac_conv, cpadded, mode='same')
            ac_conv = ac_conv if ac_conv.shape[0] == olen else ac_conv[olen // 2:-olen // 2]

    # vector sum, keeps the values only but don't keep order
    nc_sum = nc_vecs[0]
    for v in nc_vecs[1:]:
        nc_sum = np.add(nc_sum, v)

    ac_sum = ac_vecs[0] if len(ac_vecs) > 0 else codebook[0]
    if len(ac_vecs) > 1:
        for v in ac_vecs[1:]:
            ac_sum = np.add(ac_sum, v)

    # case representation -> dim = 3
    islower_case = c.islower()
    isupper_case = c.isupper()
    notcase = not (c.lower() or c.upper())  # only true if is not all upper or lower
    # starts with uppercase or not -> dim = 2 10|01
    istitle = c.istitle()
    # if all elements are numeric (does not understand decimals) -> dim = 3
    isnum = c.isnumeric()  # takes into account other things like exponents, japanese and chinese numeric characters
    isalnum = c.isalnum()
    isalpha = c.isalpha()

    code_dict = {
        'token': c,  # Normalized NFKD token
        'complete_conv': nc_conv,
        'non_accent_conv': ac_conv,
        'complete_sum': nc_sum,
        'non_accent_sum': ac_sum,
        'casing': [isupper_case, islower_case, notcase, istitle],
        'alnum': [isnum, isalnum, isalpha],
        'len': c_len,  # length -> I can encode it with Fourier approximations, a few sine waves should suffice
    }

    return code_dict


# CHAR_FPATH = "/home/leo/projects/Datasets/text/wiki-unicode/selected_sources_small/selected_chars.chars"
CHAR_FPATH = "./charsets/selected_chars.chars"


def compositional_code_main(fpath=CHAR_FPATH, reserved_codespace=RESERVED_CODE_SPACE, size_factor=2):
    """

    :param fpath: path to the char vocabulary
    :param reserved_codespace: code spaces to NOT touch, reserved
    :param size_factor: the size factor for the circular convolution composition
    :return: dictionary of the codes
    """

    # recover source of the chars to encode
    with open(fpath, "r") as f:
        chars = f.read()
    # complete the symbols checking that there are upper and lowercase, ensure that they are encoded in
    # the right normalization
    all_chars = []
    first_symbols = []
    # print("chars len = ", len(chars))
    # print("1 first_symbols len = ", len(first_symbols))
    for c in chars:
        # need to normalize the basic code to avoid later normalization mismatch with NFKD
        cc = unicodedata.normalize('NFKC', c)
        all_chars.append(cc.upper())
        all_chars.append(cc.lower())
        nc = unicodedata.normalize('NFKD', c)
        for i in nc:
            # be sure all cases are represented in the set
            first_symbols.append(i.upper())
            first_symbols.append(i.lower())
            break
    # print("1 all_chars len = ", len(all_chars))
    all_chars = sorted(list(set(all_chars)))
    # print("2 all_chars len = ", len(all_chars))
    # print("2 first_symbols len = ", len(first_symbols))
    # sort and take out a few symbols that I don't want and couldn't set in the filter of the original file
    first_symbols = sorted(list(set(first_symbols).difference(set(['҈',  '҉']))))
    # print("3 first_symbols len = ", len(first_symbols))
    all_base_chars = sorted(list(set(first_symbols + list(unicodedata.normalize('NFKD', ''.join(chars))))))
    # print("all_base_chars len = ", len(all_base_chars))

    # create the base codebook from which the composition will be created
    codes, char2int, int2char = create_base_codebook(all_base_chars, code_size=len(all_base_chars) + reserved_codespace,
                                                     # N=24, k=3, subcode_list=(2, 5, 7, 11, 13)
                                                     N=22, k=3, subcode_list=(2, 5, 7, 11, 13)
                                                     )
    # print("all_base_chars len = ", len(all_base_chars))

    # create the base matrices for the compositions
    codematrix = np.concatenate(codes, axis=1).astype('float16')
    # padding for circular convolution
    padded_codematrix = np.zeros((codematrix.shape[0], codematrix.shape[1] * size_factor)).astype('float16')
    pad_dim = (padded_codematrix.shape[1] - codematrix.shape[1]) // 2
    # pad_dim = codematrix.shape[1] // 2
    padded_codematrix[:, pad_dim:-pad_dim] = codematrix
    # circular padding to make circular convolution a reality with numpy.convolve
    # is done here to do it only once and with matrix operations
    circ_padded_codematrix = np.concatenate([padded_codematrix, padded_codematrix], axis=1)

    # create now all the charcodes dictionaries from which all compositional codes will be derived
    charcodes = [get_code_item(c, codematrix, padded_codematrix, circ_padded_codematrix, char2int) for c in all_chars]
    # print("charcodes len = ", len(charcodes))
    # charcodes now can be used as a database
    return charcodes


def charcodes_dict2codebook(charcodes, fields=('complete_conv', 'non_accent_sum', 'casing', 'alnum'), dtype='int8'):
    """
    Converts a list of charcode dicts to a codebook and assignation mapping dicts for the encoding to be used

    :param charcodes: a list containing the charcodes with the format output of get_code_item function
    :param fields: the fields to use for the final codebook charcode MUST be the among the following fields:
        'complete_conv', 'non_accent_conv', 'complete_sum', 'non_accent_sum', 'casing', 'alnum', 'len'
    :param dtype: datatype for the output codebook. Default int8 as we don't need more for small convolutions

    :return: a tuple (codebook, symbol2int, int2symbol)
    """
    charcodes = sorted(charcodes, key=lambda k: k['token'])
    codes = []
    symbol2int = OrderedDict()
    int2symbol = OrderedDict()
    for i in range(len(charcodes)):
        c = charcodes[i]
        symbol = c['token']
        symbol2int[symbol] = i
        int2symbol[i] = symbol
        vecs = []
        for f in fields:
            vecs.append(np.array(c[f], dtype=dtype))
        code = np.concatenate(vecs)
        codes.append(code)

    codes = np.stack(codes)
    return codes, symbol2int, int2symbol


