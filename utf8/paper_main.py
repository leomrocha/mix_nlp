from itertools import combinations
import os
import pickle
import numpy as np
import scipy as sp
import scipy.sparse
import time

try:
    from .utf8_encoder import *
except:
    # to solve issue with ipython executing this import
    from utf8_encoder import *

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
                    results.append((code_points, v, N, k, '{:.3f}'.format(k/N)))
                    break
    return results


def sparse_code_Nk(code_size, N, k):
    # get the indices for the ones
    comb = combinations(list(range(N)), k)  # iterator
    # limit to code size
    comb = np.array(list(comb))[:code_size]
    # compute referential as flat index to be able to use for
    comb = comb + np.array(range(128)).reshape([code_size, 1]) * N
    # convert indices to dense binary matrix
    sc = np.zeros([code_size, N])
    np.put(sc, comb)
    return sc


def create_sparse_codes():
    sc1seg = sparse_code_Nk(NCODES[0], 17, 2)
    sc2seg = sparse_code_Nk(NCODES[1], 24, 3)
    sc3seg = sparse_code_Nk(NCODES[2], 37, 4)
    sc4seg = sparse_code_Nk(NCODES[3], 45, 5)
    # save codes TODO
    
    return sc1seg, sc2seg, sc3seg, sc4seg


def multihot_primes():
    all_codes = []
    for i in range(2, 5):
        arr = list(combinations(PRIMES, i))
        for a in arr:
            ll = len(a)
            ncodes = np.prod(a)
            vsize = np.sum(a)
            sparsity = round(i/vsize, 3)
            all_codes.append((a, ll, sparsity, vsize, ncodes))
    all_codes = sorted(all_codes, key=lambda x: x[-2])
    codes_1seg = [i for i in all_codes if i[-1] > NCODES[0] and i[-1] < NCODES[3] * 2]
    codes_2seg = [i for i in all_codes if i[-1] > NCODES[1] and i[-1] < NCODES[3] * 2]
    codes_3seg = [i for i in all_codes if i[-1] > NCODES[2] and i[-1] < NCODES[3] * 2]
    codes_4seg = [i for i in all_codes if i[-1] > NCODES[3] and i[-1] < NCODES[3] * 2]
    return all_codes, codes_1seg, codes_2seg, codes_3seg, codes_4seg


def get_multihot_prime_code(ncodes, subcode_list):
    eyes = [np.eye(c) for c in subcode_list]
