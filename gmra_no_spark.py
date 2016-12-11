from sklearn.cluster import KMeans
import numpy as np
import incremental_svd as sv
#import mdtraj as md

class GMRA:
    
    def __init__(self, data, manifold_dim, resolution):
        self.data = np.transpose(data) #since our data is in rows
        self.dim = manifold_dim
        self.res = resolution
        self.low_dim_reps = []
        self.resolutions = []

    def split_step(self, data):
        data = np.transpose(data)
        km = KMeans(n_clusters=2, random_state=0).fit(data)
        to_split = zip(data,km.labels_)
        cluster_0 = filter(lambda (entry,cluster) : cluster == 0, to_split)
        cluster_1 = filter(lambda (entry,cluster) : cluster == 1, to_split)
        return np.transpose(np.asarray(map(lambda (entry,cluster) : entry, cluster_0))), np.transpose(np.asarray(map(lambda (entry,cluster) : entry, cluster_1)))

    def proj_matrix(self, data, dim):
        c_jk = data.mean(1)
        c_jk = np.reshape(c_jk,(c_jk.shape[0],1))
        centered_data = data - c_jk
        if centered_data.matrix.shape[1]<15000:
            Phi_jk, _, _ = np.linalg.svd(centered_data,full_matrices=False)
        else:
            Phi_jk, _, _ = sv.svd(centered_data,dim)
        return c_jk, Phi_jk[:,0:dim]

    def proj_points(self, data, c_jk, Phi_jk):
        low_dim_rep = np.transpose(Phi_jk).dot(data-c_jk)
        #Proj_matrix = Phi_jk.dot(np.transpose(Phi_jk))
        rep = Phi_jk.dot(low_dim_rep) + c_jk
        return low_dim_rep,rep

    def fit(self, data=None, dim=None, res=None):
        if data==None:
            data = self.data
        if dim==None:
            dim = self.dim
        if res==None:
            res = self.res
        c_jk = data.mean(1)
        c_jk = np.reshape(c_jk,(c_jk.shape[0],1))
        centered_data = data-c_jk
        #Phi_jk, _, _ = np.linalg.svd(centered_data,full_matrices=False)
        Phi_jk, _, _ = sv.svd(centered_data,dim)
        resolutions = [(data,c_jk,Phi_jk[:,0:dim])]
        low_dim_reps = [np.transpose(Phi_jk[:,0:dim]).dot(data)]
        for j in xrange(2**res+1):
            print "At "+str(j)
            if resolutions[j] != None and resolutions[j][0].shape[1]>1:
                cluster_0,cluster_1 = self.split_step(resolutions[j][0])
                c_jk0, Phi_jk0 = self.proj_matrix(cluster_0,dim)
                c_jk1, Phi_jk1 = self.proj_matrix(cluster_1,dim)
                if self.subsp_angle(Phi_jk0,Phi_jk1) < 0.99999: #replace by epsilon of choice
                    low_dim_rep_k0,rep_k0 = self.proj_points(cluster_0,c_jk0,Phi_jk0)
                    low_dim_rep_k1,rep_k1 = self.proj_points(cluster_1,c_jk1,Phi_jk1)
                    resolutions += [(rep_k0,c_jk0,Phi_jk0),(rep_k1,c_jk1,Phi_jk1)]
                    low_dim_reps += [low_dim_rep_k0,low_dim_rep_k1]
                else:
                    resolutions += [None]
                    low_dim_reps += [None]
        self.low_dim_reps = low_dim_reps
        self.resolutions = resolutions
        return low_dim_reps,resolutions

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

    
            
    
            
            
        
    
    
        
