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
        for N in range(32, 128):
            for k in [3, 4, 5, 6]:
                v = np.prod(list(range(N, N - k, -1))) / np.prod(list(range(1, k + 1)))
                if v > code_points:
                    # print("code_size={}; N={},k={}".format(v, N, k))
                    results.append((code_points, v, N, k))
    return results


def sparse_code_Nk(code_size, N, k):
    ret = combinations(list(range(N)), k)  # iterator
    ret = np.array(list(ret))[:code_size]
    return ret


def multihot_primes():
    codes_1seg = []
    codes_2seg = []
    codes_3seg = []
    codes_4seg = []
    codes = []
    for i in range(2, 7):
        arr = np.array(list(combinations(PRIMES, i)))
        ncodes = np.prod(arr, axis=1).reshape(-1, 1)
        vsizes = np.sum(arr, axis=1).reshape((-1, 1))
        # print(arr.shape, ncodes.shape)
        arr = np.hstack([arr, vsizes, ncodes])
        arr = arr[arr[:, 2].argsort()]
        codes.append(arr)
        # Now filter the alternatives that can handle the space needed for each code
        # NCODES
        arr_1s = arr[arr[:, -2] > NCODES[0]]
        arr_2s = arr[arr[:, -2] > NCODES[1]]
        arr_3s = arr[arr[:, -2] > NCODES[2]]
        arr_4s = arr[arr[:, -2] > NCODES[3]]
        codes_1seg.append(arr_1s)
        codes_2seg.append(arr_2s)
        codes_3seg.append(arr_3s)
        codes_4seg.append(arr_4s)
    return codes, codes_1seg, codes_2seg, codes_3seg, codes_4seg
#     return np.array(codes)
