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
        mb = os.path.getsize(name) / (1024**2)
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
