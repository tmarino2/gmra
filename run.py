import mdtraj as md
import numpy as np
import gmra_no_spark as gm

for j in range(1,10):
    file_name = ''
    topology_name = '/mddb2/md/bpti-prot/bpti-prot.pdb'
    if j < 10:
        file_name = '/mddb2/md/bpti-prot/bpti-prot-0%d.dcd'%(j)
    else:
        file_name = '/mddb2/md/bpti-prot/bpti-prot-%d.dcd'%(j)
    traj = md.load(file_name, top=topology_name)
    traj = traj.xyz[0:80000]
    traj = np.reshape(traj,(80000,892*3))
    gmra = gm.GMRA(traj,10,7)
    resolutions, low_dim_rep = gmra.fit()
    rep_k = open('rep_k2','a+')
    c_jk = open('c_jk2','a+')
    Phi_jk = open('Phi_jk2','a+')
    low_dim_reps = open('low_dim_reps2','a+')
    for i in xrange(len(res)):
        rep_k_l = 'representation %d\n'%(i)
        c_jk_l = 'c_jk %d\n'%(i)
        Phi_jk_l = 'Phi_jk_l %d\n'%(i)
        low_dim_reps_l = 'low_dim_reps %d\n'%(i)
        rep_k.write(rep_k_l)
        c_jk.write(c_jk_l)
        Phi_jk.write(Phi_jk_l)
        low_dim_reps.write(low_dim_reps_l)
        np.savetxt(rep_k,gmra.resolutions[i][0])
        np.savetxt(c_jk,gmra.resolutions[i][1])
        np.savetxt(Phi_jk,gmra.resolutions[i][2])
        np.savetxt(low_dim_reps,gmra.low_dim_reps[i])
    rep_k.close()
    c_jk.close()
    Phi_jk.close()
    low_dim_reps.close()
