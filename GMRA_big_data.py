import numpy as np
from pyspark.mllib.linalg.distributed import *
from pyspark.mllib.clustering import *
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
from pyspark.mllib.linalg.distributed import *

class SVD(JavaModelWrapper):
    """Wrapper around the SVD scala case class"""
    @property
    def U(self):
        """ Returns a RowMatrix whose columns are the left singular vectors of the SVD if computeU was set to be True."""
        u = self.call("U")
        if u is not None:
            return RowMatrix(u)
    @property
    def s(self):
        """Returns a DenseVector with singular values in descending order."""
        return self.call("s")
    @property
    def V(self):
        """ Returns a DenseMatrix whose columns are the right singular vectors of the SVD."""
        return self.call("V")

def computeSVD(row_matrix, k, computeU=False, rCond=1e-9):
    """
    Computes the singular value decomposition of the RowMatrix.
    The given row matrix A of dimension (m X n) is decomposed into U * s * V'T where
    * s: DenseVector consisting of square root of the eigenvalues (singular values) in descending order.
    * U: (m X k) (left singular vectors) is a RowMatrix whose columns are the eigenvectors of (A X A')
    * v: (n X k) (right singular vectors) is a Matrix whose columns are the eigenvectors of (A' X A)
    :param k: number of singular values to keep. We might return less than k if there are numerically zero singular values.
    :param computeU: Whether of not to compute U. If set to be True, then U is computed by A * V * sigma^-1
    :param rCond: the reciprocal condition number. All singular values smaller than rCond * sigma(0) are treated as zero, where sigma(0) is the largest singular value.
    :returns: SVD object
    """
    java_model = row_matrix._java_matrix_wrapper.call("computeSVD", int(k), computeU, float(rCond))
    return SVD(java_model)

class C_jk:
    #transpose the rdd in the init and then do all operations on the column-wise data 
    def __init__(self,C_jk,dim):#replace with path to file where each C_jk is stored
        self.C_jk = self.rddTranspose(C_jk)
        self.c_jk = None
        self.P_jk = None
        self.dim = dim
        self.C_jk_projected = None
        
    def rddTranspose(self,rdd):
        rddT1 = rdd.zipWithIndex().flatMap(lambda (x,i): [(i,j,e) for (j,e) in enumerate(x)])
        rddT2 = rddT1.map(lambda (i,j,e): (j, (i,e))).groupByKey().sortByKey()
        rddT3 = rddT2.map(lambda (i, x): sorted(list(x),cmp=lambda (i1,e1),(i2,e2) : cmp(i1, i2)))
        rddT4 = rddT3.map(lambda x: map(lambda (i, y): y , x))
        return rddT4.map(lambda x: np.asarray(x))

    def mean(self):
        self.c_jk = self.C_jk.map(np.mean)
        return np.asarray(self.c_jk.collect())
    
    def compute_proj(self):
        if self.c_jk == None:
            self.mean()
        mean_zip = self.C_jk.zip(self.c_jk)
        C_jk_centered = mean_zip.map(lambda (row,m): np.asarray(map(lambda row_el: row_el - m, row)))
        svd = computeSVD(RowMatrix(C_jk_centered),self.dim,computeU=True)
        P_jk = svd.U.rows
        self.P_jk = np.asarray(P_jk.map(lambda row: row.toArray()))
        return self.P_jk

    def project(self):
        if self.P_jk == None:
            self.compute_proj()
        P_jk_T = np.transpose(self.P_jk)
        self.C_jk_projected = self.C_jk.map(lambda point: np.transpose(self.P_jk).dot(point))
        return self.C_jk_projected

