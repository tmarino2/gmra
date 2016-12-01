import numpy as np
from pyspark.mllib.linalg.distributed import *
from pyspark.mllib.clustering import *
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
from pyspark.mllib.linalg.distributed import *
'''Todo: clear code up (check for inconsistent dimensions in operations, etc.), implement wavelets, find a faster matrix mult?, wrap everything into a class'''

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


def add(rdd1,rdd2):
    #takes two rdds from a numpy_matrix the same shape and returns their sum
    if rdd1.count() != rdd2.count():
        raise Exception("Number of rows missmatch!")
    if len(rdd1.first()) != len(rdd2.first()):
        raise Exception("Number of cols missmatch!")
    mat3 = mat1.zip(mat2)
    return mat3.map(lambda (arr1,arr2) : arr1+arr2)

def norm_square(rdd1,rdd2):
    #Frobenius norm
    rdd = add(rdd1,mult_by_sc(rdd2,-1))
    return np.linalg.norm(rdd.map(lambda row: np.linalg.norm(row)).collect())

def mult_matr(mat1,mat2):
    #takes two rdds mat1 mxn and mat2 nxm and forms the matrix product
    #mat1*mat2 in mxm
    mat2 = rddTranspose(mat2)
    m = mat1.count()
    mat_cart = mat1.cartesian(mat2)
    mat_to_be_reshaped = mat_cart.map(lambda (arr1,arr2) : sum(map(lambda (x,y) : x*y, zip(arr1,arr2)))).zipWithIndex() #long rdd of m^2 entries where each entry is a scalar
    return mat_to_be_reshaped.groupBy(lambda (x,i): i/m).map(lambda row : list(row)).map(lambda (idx,row) : np.asarray([x for (x,i) in list(row)]))

def rddTranspose(rdd):
    rddT1 = rdd.zipWithIndex().flatMap(lambda (x,i): [(i,j,e) for (j,e) in enumerate(x)])
    rddT2 = rddT1.map(lambda (i,j,e): (j, (i,e))).groupByKey().sortByKey()
    rddT3 = rddT2.map(lambda (i, x): sorted(list(x),cmp=lambda (i1,e1),(i2,e2) : cmp(i1, i2)))
    rddT4 = rddT3.map(lambda x: map(lambda (i, y): y , x))
    return rddT4.map(lambda x: np.asarray(x))
#Taken from http://www.data-intuitive.com/2015/01/transposing-a-spark-rdd/

def mult_by_sc(rdd,scalar):
    #multiplies by scalar
    return rdd.map(lambda arr : map(lambda arr_el : scalar*arr_el, arr))

def center_data_matr(rdd):
    #take an rdd containing points row-wise
    #return a meme-centered rdd which is tranposed
    rdd_t = rddTranspose(rdd)
    meme_zipped = rdd_t.zip(rdd_t.map(np.mean))
    meme_centered = meme_zipped.map(lambda (row,m): np.asarray(map(lambda row_el: row_el - m, row)))
    return rddTranspose(meme_centered)

def meme_of_columns(rdd):
    return rdd.map(np.mean)

def substr_vec(rdd,vec):
    #takes an rdd with each entry a single float
    #substracts the vector from all of the columns of the given rdd
    rdd_t = rdd
    meme_zipped = rdd_t.zip(vec)
    meme_centered = meme_zipped.map(lambda (row,m): np.asarray(map(lambda row_el: row_el - m, row)))
    return meme_centered

def cluster_step(rdd):
    #take an rdd with row-wise data and split it in two clusters with k-means
    #return the two rdd's wrt to the clusters
    kmeans_b = KMeans.train(rdd, 2, maxIterations=10, seed=50, initializationSteps=5)
    centers = kmeans_b.predict(rdd)
    to_split = rdd.zip(centers)
    #probably a better way to do this
    cluster_0 = to_split.filter(lambda (entry,cluster) : cluster == 0)
    cluster_1 = to_split.filter(lambda (entry,cluster) : cluster == 1)
    return cluster_0.map(lambda (entry,cluster) : entry), cluster_1.map(lambda (entry,cluster) : entry)

def proj_step(rdd,d):
    #takes as input partition j,k and manifold dimension d
    #outputs projection matrix \Phi_j,k and center c_j,k
    rdd_t = rddTranspose(rdd)
    c_jk = meme_of_columns(rdd_t)
    centered_rdd = substr_vec(rdd_t,c_jk)
    svd = computeSVD(RowMatrix(centered_rdd),d,computeU=True)
    Phi_jk = svd.U.rows #hopefully this has orthogonal columns and not rows
    return c_jk, Phi_jk.map(lambda row: row.toArray())

def proj_points(rdd,c_jk,Phi_jk):
    #input rdd and outputs are row-wise
    #takes an rdd of points to be projected in M_jk
    #returns to low d-dimensional representation in M_jk,
    #the n-dimensional orthogonal projection onto M_jk
    low_dim_rep = mult_matr(rdd,Phi_jk) #rdd is n x dim Phi_jk is dim x d
    Proj_matrix = mult_matr(Phi_jk,rddTranspose(Phi_jk))
    rdd_t = rddTranspose(rdd)
    rep = mult_matr(Proj_matrix,substr_vec(rdd_t,c_jk))
    rep = substr_vec(rep,mult_by_sc(c_jk,-1))
    return low_dim_rep,rddTranspose(rep)

def gmra(rdd,resolution,d):
    #children of j-th entry are at 2**j+1 and 2**j+2
    #parent of j-th entry is at log(j-1(or 2)) dependng if j is odd or even
    c_jk,Phi_jk = proj_step(rdd,d)
    resolutions = [(rdd,c_jk,Phi_jk)]
    low_dim_rep = [proj_points(rdd,c_jk,Phi_jk)[0]]
    '''rep = [rdd]
    proj_matrices = [sc.parallelize(np.eye(rdd.first().shape[0]))]
    centers = [c_jk]
    clusters = [rdd]'''
    #wavelets = []
    for j in xrange(2**resolution+1):
        cluster_0,cluster_1 = cluster_step(clusters[j])
        #clusters += [cluster_0,cluster_1]
        c_jk, Phi_jk = proj_step(cluster_0,d)
        c_jk1, Phi_jk1 = proj_step(cluster_1,d)
        #centers += [c_jk,c_jk1]
        ldr_k,rep_k = proj_points(cluster_0,c_jk,Phi_jk)
        ldr_k1,rep_k1 = proj_points(cluster_1,c_jk1,Phi_jk1)
        resolutions += [(rep_k,c_jk,Phi_jk),(rep_k1,c_jk1,Phi_jk1)]
        low_dim_rep += [ldr_k,ldr_k1]
        #rep += [rep_k,rep_k1]
        #proj_matrices += [pr_k,pr_k1]
    #return low_dim_rep,rep,centers,proj_matrices,clusters
    return low_dim_reps,resolutions
