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
            (2, NCODES[1], (24, 3), (3, 5, 11, 13), (6, 2), 64, 9/64),
            # (2, 1916, (24, 3), (3, 5, 11, 13), (6, 2), 64, 9 / 64),
            (3, NCODES[2], (37, 4), (11, 13, 19, 23), (11, 7, 4, 3), 128, 12/128),
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
CONFIG = (2, 2112, (24, 3), (3, 5, 11, 13), (4, 6, 8, 10, 12), 96, 13 / 96)
OFNAME = "codes/adhoc-codebook-2112.pkl"


def create_codebook(charset_fpath=CHARSET_PATH, config=CONFIG,
                    ofname=OFNAME,
                    special_codes=SPECIAL_CODES,
                    nul_row_is_zero=True,
                    reserved_spaces=32
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
        # could use UNK but I'd rather have it be obviously different
        int2char[i] = UNASSIGNED
    # overwrite the indices of the reverse mapping for the special codes
    for c, i in special_codes:
        # Take into account this will duplicate the char2int mapping having 2 chars going to the same int
        char2int[c] = i
        # but the int reverse index will be overwritten
        int2char[i] = c
    with open(charset_fpath, 'r') as f:
        all_chars = f.read()
        for i, c in enumerate(all_chars):
            # forward the index
            i = i + reserved_spaces
            char2int[c] = i
            int2char[i] = c

    # pickle all together
    codebook = (codes, char2int, int2char)
    with open(ofname, 'wb') as f:
        print("saving file {} with codes.shape {} | char2int {} | int2char {}".format(
            ofname, codes.shape, len(char2int), len(int2char)))
        pickle.dump(codebook, f, pickle.HIGHEST_PROTOCOL)

    return codebook
