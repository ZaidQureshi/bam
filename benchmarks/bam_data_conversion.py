#script for data conversation

import numpy as np
from scipy.sparse import csr_matrix
import scipy.io

def mtx_to_bel(mtx_file_path):
    # read mtx
    data = scipy.io.mmread(mtx_file_path)

    # convert into csr_matrix
    csr_matrix_data = data.tocsr()

    # extract CSR data
    indptr = csr_matrix_data.indptr.astype(np.int64) # .bel.col
    indices = csr_matrix_data.indices.astype(np.int64) # .bel.dst
    data = csr_matrix_data.data.astype(np.float32) # .bel.val
    nnz = csr_matrix_data.nnz # Total number of non-0 elements

    # save path
    belfile_colpath = "MOLIERE_2016.bel.col"
    belfile_dstpath = "MOLIERE_2016.bel.dst"
    belfile_valpath = "MOLIERE_2016.bel.val"

    # write to bel
    with open(belfile_colpath, "wb") as f:
        # Header(nnz + placeholder)
        np.array([nnz, 0], dtype=np.int64).tofile(f)
        indptr.tofile(f) # .bel.col
    with open(belfile_dstpath, "wb") as f:
        np.array([nnz, 0], dtype=np.int64).tofile(f)
        indices.tofile(f) # .bel.dst
    with open(belfile_valpath, "wb") as f:
        np.array([nnz, 0], dtype=np.int64).tofile(f)
        data.tofile(f) # .bel.val

if __name__ == "__main__":
    mtx_to_bel('MOLIERE_2016.mtx') #datasetname

