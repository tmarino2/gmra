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
    return low_dim_rep,rddTranspose(rep),Proj_matrix

'''def compute_wavelet(rdd,c_jk,Phi_jk,c_j2k,Phi_j1k):
    #compute the wavelet Q_j+1,k between j+1,k and j,k
    #i.e. x_j+1 - x_j = Q_j+1,k(x)
    #wavelet is row-wise
    _, rep_j1k = proj_points(rdd,c_j1k,Phi_j1k)
    _, rep_jk = proj_points(rdd,c_jk,Phi_jk)
    return add(rep_j1l,mult_by_sc(rep_jk,-1))'''

def gmra(rdd,resolution,d):
    #children of j-th entry are at 2**j+1 and 2**j+2
    #parent of j-th entry is at log(j-1(or 2)) dependng if j is odd or even
    c_jk = meme_of_columns(rddTranspose(rdd))    
    low_dim_rep = [rdd]
    rep = [rdd]
    proj_matrices = [sc.parallelize(np.eye(rdd.first().shape[0]))]
    centers = [c_jk]
    clusters = [rdd]
    #wavelets = []
    for j in xrange(2**resolution):
        cluster_0,cluster_1 = cluster_step(clusters[j])
        clusters += [cluster_0,cluster_1]
        c_jk, Phi_jk = proj_step(cluster_0,d)
        c_jk1, Phi_jk1 = proj_ste(cluster_1,d)
        centers += [c_jk,c_jk1]
        ldr_k,rep_k,pr_k = proj_points(cluster_0,c_jk,Phi_jk)
        ldr_k1,rep_k1,pr_k1 = proj_points(cluster_1,c_jk1,Phi_jk1)
        low_dim_rep += [ldr_k,ldr_k1]
        rep += [rep_k,rep_k1]
        proj_matrices += [pr_k,pr_k1]
        
    
