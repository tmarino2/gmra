import numpy as np

def incremental_step(U,Sigma,V,v_t,k):
    #partial svd of matrix, V is transposed, v_t is column to be added, k are components
    w_t = np.transpose(U).dot(v_t)
    print "w_t shape",w_t.shape
    p_t = U.dot(w_t)
    r_t = v_t - p_t
    n = Sigma.shape[1]
    S = np.vstack((Sigma,np.zeros((1,n))))
    w = np.vstack((w_t.reshape(w_t.shape[0],1),np.linalg.norm(r_t)))
    print "w shape, S shape",w.shape,S.shape
    S = np.hstack((S,w))
    U_cap,Sigma_cap,V_cap = np.linalg.svd(S,full_matrices = True)
    r = r_t*0
    if np.linalg.norm(r_t) > 0.00001:
        r = r_t/np.linalg.norm(r_t)
    print "U shape r shape",U.shape,r.shape
    U_t1 = np.hstack((U,r.reshape(r.shape[0],1))).dot(U_cap)
    Sigma_t1 = np.diag(Sigma_cap)
    one = np.zeros((V_cap.shape[0],1))
    one[-1:,:] = 1
    V_t1 = V_cap.dot(np.hstack((np.vstack((V,np.zeros((V.shape[1])))),one)))
    '''else:
        print "U shape, U_cap shape", U.shape,U_cap.shape
        U_t1 = U.dot(U_cap[:-1,:-1])
        Sigma_t1 = np.diag(Sigma_cap)[:-1,:-1]
        V_t1 = V_cap[:,:-1].dot(V)'''
    if len(Sigma_cap) >= k or np.linalg.norm(r_t) > 0.00001:
        U_t1 = U_t1[:,:-1]
        Sigma_t1 = Sigma_t1[:-1,:-1]
        V_t1 = V_t1[:-1,:]
    print U_t1.shape,Sigma_t1.shape,V_t1.shape
    return U_t1,Sigma_t1,V_t1

def svd(X,k):
    #column-wise data, number of components
    U,s,V = np.linalg.svd(X[:,:k],full_matrices = False)
    S = np.diag(s)
    for i in range(k,X.shape[1]):
        U,S,V = incremental_step(U,S,V,X[:,i],k)
    return U,S,V
    
