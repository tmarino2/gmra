import mdtraj as md
import numpy as np
import gmra_no_spark as gm
import sys
import argparse

argsp = argparse.ArgumentParser()
argsp.add_argument('--num_points', type=int, default=80000)
argsp.add_argument('--manifold_dim', type=int, default=10)
argsp.add_argument('--resolution', type=int, default=7)
args = argsp.parse_args()

topology_name = '/mddb2/md/bpti-prot/bpti-prot.pdb'
align_to = md.load('/mddb2/md/bpti-prot/bpti-prot-00.dcd',top=topology_name)

for j in range(0,10):
    file_name = ''
    save_file = 'bpti-prot-%d.b'%(j)
    if j < 10:
        file_name = '/mddb2/md/bpti-prot/bpti-prot-0%d.dcd'%(j)
    else:
        file_name = '/mddb2/md/bpti-prot/bpti-prot-%d.dcd'%(j)
    traj = md.load(file_name, top=topology_name)
    traj.superpose(align_to)
    traj = traj.xyz[0:args.num_points]
    traj = np.reshape(traj,(args.num_points,892*3))
    gmra = gm.GMRA(traj,args.manifold_dim,args.resolution)
    ldr,res = gmra.fit()
    gmra.save_model(save_file)
