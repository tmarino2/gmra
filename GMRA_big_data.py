import numpy as np
from sklearn.cluster import KMeans as kmeans
import mdtraj as md
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg.distributed import *
from pyspark.mllib.clustering import *
from pyspark.mllib.common import callMLlibFunc, JavaModelWrapper
from pyspark.mllib.linalg.distributed import *

#conf = SparkConf().setAppName("GMRA")
#sc = SparkContext(conf=conf)
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
        #self.C_jk = self.rddTranspose(C_jk)
        self.C_jk = C_jk
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
        self.c_jk = self.C_jk.reduce(lambda arr1,arr2: arr1+arr2)
        #c_jk = np.copy(self.c_jk)
        #self.c_jk = sc.parallelize(c_jk)
        #self.c_jk = self.C_jk.map(np.mean)
        return self.c_jk
    
    def compute_proj(self):
        if self.c_jk == None:
            self.mean()
        #mean_zip = self.C_jk.zip(self.c_jk)
        #C_jk_centered = mean_zip.map(lambda (row,m): np.asarray(map(lambda row_el: row_el - m, row)))
        c_jk = np.copy(self.c_jk)#fucking python lazy evaluation
        C_jk_centered = self.C_jk.map(lambda row: row - c_jk)
        #C_jk_centered.first()#fucking python lazy evaluation
        rowMatr = RowMatrix(C_jk_centered)
        svd = computeSVD(rowMatr,self.dim)
        self.P_jk = svd.V.toArray()
        return self.P_jk
    
    def project(self):
        if self.P_jk == None:
            self.compute_proj()
        P_jk = np.copy(self.P_jk)#fucking python lazy evaluation
        C_jk_projected = self.C_jk.map(lambda point: np.transpose(P_jk).dot(point))
        self.C_jk_projected = C_jk_projected
        return C_jk_projected
    
class GMRA:
    
    def __init__(self, data, manifold_dim, resolution, memory, sc):
        #self.data = np.transpose(data) #since our data is in rows
        self.data = data
        self.dim = manifold_dim
        self.res = resolution
        self.resolutions = []
        self.mem = memory
        self.sc = sc
        
    def split_step_rdd(self,rdd):
        #take an rdd with row-wise data and split it in two clusters with k-means
        #return the two rdd's wrt to the clusters
        kmeans_b = KMeans.train(rdd, 2, maxIterations=10, seed=50, initializationSteps=5)
        centers = kmeans_b.predict(rdd)
        to_split = rdd.zip(centers)
        #probably a better way to do this
        cluster_0 = to_split.filter(lambda (entry,cluster) : cluster == 0)
        cluster_1 = to_split.filter(lambda (entry,cluster) : cluster == 1)
        return cluster_0.map(lambda (entry,cluster) : entry), cluster_1.map(lambda (entry,cluster) : entry)
    
    def split_step(self, data):
        km = kmeans(n_clusters=2, random_state=0).fit(data)
        to_split = zip(data,km.labels_)
        cluster_0 = filter(lambda (entry,cluster) : cluster == 0, to_split)
        cluster_1 = filter(lambda (entry,cluster) : cluster == 1, to_split)
        return np.asarray(map(lambda (entry,cluster) : entry, cluster_0)), np.asarray(map(lambda (entry,cluster) : entry, cluster_1))
    
    def project_cluster(self, data):
        C = C_jk(data,self.dim)
        low_dim_rep = C.project()
        return C.c_jk, C.P_jk, low_dim_rep
    
    def proj_matrix(self, data):
        data = np.transpose(data)
        c_jk = data.mean(1)
        c_jk = np.reshape(c_jk,(c_jk.shape[0],1))
        centered_data = data - c_jk
        Phi_jk, _, _ = np.linalg.svd(centered_data,full_matrices=False)
        low_dim_rep = np.transpose(Phi_jk[:,0:self.dim]).dot(centered_data)
        return c_jk, Phi_jk[:,0:self.dim], np.transpose(low_dim_rep)
        
    def next_res_rdd(self, res_jk):
        resolutions = []
        if res_jk != None and res_jk[0].count()>1:
            cluster_0,cluster_1 = self.split_step_rdd(res_jk[0])
            c_jk0, Phi_jk0, low_dim_rep_k0 = self.project_cluster(cluster_0)
            c_jk1, Phi_jk1, low_dim_rep_k1 = self.project_cluster(cluster_1)
            if self.subsp_angle(Phi_jk0,Phi_jk1) < 0.99999: #replace by epsilon of choice
                resolutions = [(cluster_0,c_jk0,Phi_jk0,low_dim_rep_k0),(cluster_1,c_jk1,Phi_jk1,low_dim_rep_k1)]
            else:
                resolutions = [None]
                #low_dim_reps = [None]
        self.resolutions += resolutions
        return resolutions
        
    def next_res_sub(self, res_jk, dim):
        #something is not right here
        resolutions = []
        if res_jk != None and len(res_jk[0])>1:
            cluster_0, cluster_1 = self.split_step(res_jk[0])
            print cluster_0.shape,cluster_1.shape
            c_jk0, Phi_jk0, low_dim_rep_k0 = self.proj_matrix(cluster_0)
            c_jk1, Phi_jk1, low_dim_rep_k1 = self.proj_matrix(cluster_1)
            if self.subsp_angle(Phi_jk0,Phi_jk1) < 0.99999:
                resolutions = [(cluster_0,c_jk0,Phi_jk0,low_dim_rep_k0),(cluster_1,c_jk1,Phi_jk1,low_dim_rep_k1)]
            else:
                resolutions = [None,None]
        return resolutions
        
    def next_res(self, rdd_j):
        rdd_j1 = rdd_j.flatMap(lambda res_jk: next_res_sub(res_jk,self.dim))
        print "next_res rdd size",rdd_j1.count()
        self.resolutions += rdd_j1.collect()
        return rdd_j1

    def compute_mem(self, rdd):
        row_size = (len(rdd.first())*rdd.first().itemsize)/1024.0 #approximate size of row in KBs
        return rdd.count()*row_size #make this approximately compute memory taken by rdd
    
    def fit(self, data=None, dim=None, res=None, mem=None):
        if data==None:
            data = self.data
        if dim==None:
            dim = self.dim
        if res==None:
            res = self.res
        c_jk, Phi_jk, low_dim_rep = self.project_cluster(data)
        self.resolutions = [(data,c_jk,Phi_jk,low_dim_rep)]
        fit_in_mem = False
        rdd_j = self.sc.emptyRDD()
        i = 0
        while i<=res:
            if not(fit_in_mem):
                max_mem = 0
                for j in xrange(2**i):
                    print "At resolution "+str(i)+" step "+str(j)
                    idx = 2**i-1+j
                    resolutions = self.next_res_rdd(self.resolutions[idx])
                    if max_mem < max(self.compute_mem(resolutions[0][0]),self.compute_mem(resolutions[1][0])):
                        max_mem =  max(self.compute_mem(resolutions[0][0]),self.compute_mem(resolutions[1][0]))
                print "maxmem ",max_mem
                if 4*max_mem < self.mem:
                    fit_in_mem = True
                    print "fit in mem true"
                    for j in xrange(2**(i+1)):
                        idx = 2**(i+1)-1+j
                        C_jk = np.copy(np.asarray(self.resolutions[idx][0].collect()))
                        Projected_jk = np.copy(np.asarray(self.resolutions[idx][3].collect()))
                        print "C_jk shape: ",C_jk.shape
                        temp_rdd = self.sc.parallelize([(C_jk,self.resolutions[idx][1],self.resolutions[idx][2],Projected_jk)])
                        print temp_rdd.count()
                        rdd_j = rdd_j.union(temp_rdd)
                    print "rdd_j size: ",rdd_j.count()
                    #print rdd_j.first()
            else:
                print "in mem"
                #temp_rdd = self.next_res(rdd_j)
                temp_rdd,resolutions = next_res(rdd_j,self.dim)
                print temp_rdd.count()
                self.resolutions += resolutions
                rdd_j = temp_rdd
            i+=1        
        return self.resolutions
    
    def project_test_point(self, point):
        point_reps = [np.transpose(self.resolutions[0][2]).dot(point)]
        j = 0
        while 2**j+1 < len(self.resolutions):
            if self.resolutions[j] == None:
                break
            idx1 = 2**j+1
            idx2 = 2**j+2
            dist1 = np.linalg.norm(self.resolutions[idx1][1]-point)
            dist2 = np.linalg.norm(self.resolutions[idx2][1]-point)
            if dist1 <= dist2:
                point_reps.append(np.transpose(self.resolutions[idx1][2]).dot(point-self.resolutions[idx1][1]))
                j = idx1
            else:
                point_reps.append(np.transpose(self.resolutions[idx2][2]).dot(point-self.resolutions[idx1][1]))
                j = idx2
        return point_reps
    
    def subsp_angle(self, A, B):
        ab = np.trace(np.transpose(A).dot(B))
        a = np.sqrt(np.trace(np.transpose(A).dot(A)))
        b = np.sqrt(np.trace(np.transpose(B).dot(B)))
        return ab/(a*b)

def next_res(rdd_j,dim):
    rdd_j1 = rdd_j.flatMap(lambda res_jk: next_res_sub(res_jk,dim))
    print "next_res rdd size",rdd_j1.count()
    resolutions = rdd_j1.collect()
    return rdd_j1,resolutions
    
def next_res_sub(res_jk, dim):
        #something is not right here
    resolutions = []
    if res_jk != None and len(res_jk[0])>1:
        cluster_0, cluster_1 = split_step(res_jk[0])
        print cluster_0.shape,cluster_1.shape
        c_jk0, Phi_jk0, low_dim_rep_k0 = proj_matrix(cluster_0,dim)
        c_jk1, Phi_jk1, low_dim_rep_k1 = proj_matrix(cluster_1,dim)
        if subsp_angle(Phi_jk0,Phi_jk1) < 0.99999:
            resolutions = [(cluster_0,c_jk0,Phi_jk0,low_dim_rep_k0),(cluster_1,c_jk1,Phi_jk1,low_dim_rep_k1)]
        else:
            resolutions = [None,None]
    return resolutions

def proj_matrix(data,dim):
    data = np.transpose(data)
    c_jk = data.mean(1)
    c_jk = np.reshape(c_jk,(c_jk.shape[0],1))
    centered_data = data - c_jk
    Phi_jk, _, _ = np.linalg.svd(centered_data,full_matrices=False)
    low_dim_rep = np.transpose(Phi_jk[:,0:dim]).dot(centered_data)
    return c_jk, Phi_jk[:,0:dim], np.transpose(low_dim_rep)

def split_step(data):
    km = kmeans(n_clusters=2, random_state=0).fit(data)
    to_split = zip(data,km.labels_)
    cluster_0 = filter(lambda (entry,cluster) : cluster == 0, to_split)
    cluster_1 = filter(lambda (entry,cluster) : cluster == 1, to_split)
    return np.asarray(map(lambda (entry,cluster) : entry, cluster_0)), np.asarray(map(lambda (entry,cluster) : entry, cluster_1))

def subsp_angle(A, B):
    ab = np.trace(np.transpose(A).dot(B))
    a = np.sqrt(np.trace(np.transpose(A).dot(A)))
    b = np.sqrt(np.trace(np.transpose(B).dot(B)))
    return ab/(a*b)
