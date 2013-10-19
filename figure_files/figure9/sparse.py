import csv
import numpy as np
import random as rnd
import scipy.sparse as sps
import matplotlib.pyplot as plt
#plt.spy(a, marker='.',markersize=2)
#plt.show()


def plot_sparse(mat, filename):
    plt.spy(mat, markersize=1)
    plt.tight_layout()
    #plt.rc('xtick',labelsize=16)
    #plt.rc('ytick',labelsize=16)
    plt.savefig.format = "pdf"
    plt.savefig(filename)
    plt.clf()

#--------------------------------
nrows = 512
#--------------------------------
# compact matrix.
mat = sps.csr_matrix((nrows, nrows))
ones = np.ones(2*nrows)
mat.setdiag(ones, k=0)
for i in range(1,16):
    mat.setdiag(ones, k=i)
    mat.setdiag(ones, k=-i)
plot_sparse(mat, "compact_matrix.pdf")
#--------------------------------
# supercompact matrix
ones = np.ones((nrows,32))
xones = sps.csr_matrix(ones)
xsparse = sps.csr_matrix((nrows,nrows-32))
mat1 = sps.hstack([xones,xsparse])
plot_sparse(mat1, "supercompact_matrix.pdf")
#--------------------------------
# random matrix
rg = range(nrows)
col_id = np.zeros((nrows,32),int)
for i in range(nrows):
    a = rnd.sample(rg, 32)
    col_id[i,:] = np.sort(a)

col_id = np.ravel(col_id)
values = np.ravel(np.ones((nrows,32),float))
ptrs = range(0,32*(nrows+1),32)
mat2 = sps.csr_matrix((values,col_id,ptrs),shape=(nrows,nrows))
plot_sparse(mat2, "random_matrix.pdf")
#--------------------------------
#--------------------------------
