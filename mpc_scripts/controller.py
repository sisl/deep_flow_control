# -*- coding: utf-8 -*-
import glob
import h5py
import numpy as np
import os, sys
import pickle
import subprocess
import tensorflow as tf

sys.path.append('/home/sisl/jeremy/deep_cfd/koopman')
from cvxpy import *
from koopman_model import KoopmanModel
from pyfr.mpiutil import get_comm_rank_root, get_mpi
from pyfr.plugins.base import BasePlugin


def _closest_upts_bf(etypes, eupts, pts):
    for p in pts:
        # Compute the distances between each point and p
        dists = [np.linalg.norm(e - p, axis=2) for e in eupts]

        # Get the index of the closest point to p for each element type
        amins = [np.unravel_index(np.argmin(d), d.shape) for d in dists]

        # Dereference to get the actual distances and locations
        dmins = [d[a] for d, a in zip(dists, amins)]
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Find the minimum across all element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts_kd(etypes, eupts, pts):
    from scipy.spatial import cKDTree

    # Flatten the physical location arrays
    feupts = [e.reshape(-1, e.shape[-1]) for e in eupts]

    # For each element type construct a KD-tree of the upt locations
    trees = [cKDTree(f) for f in feupts]

    for p in pts:
        # Query the distance/index of the closest upt to p
        dmins, amins = zip(*[t.query(p) for t in trees])

        # Unravel the indices
        amins = [np.unravel_index(i, e.shape[:2])
                 for i, e in zip(amins, eupts)]

        # Dereference to obtain the precise locations
        plocs = [e[a] for e, a in zip(eupts, amins)]

        # Reduce across element types
        yield min(zip(dmins, plocs, etypes, amins))


def _closest_upts(etypes, eupts, pts):
    try:
        # Attempt to use a KD-tree based approach
        yield from _closest_upts_kd(etypes, eupts, pts)
    except ImportError:
        # Otherwise fall back to brute force
        yield from _closest_upts_bf(etypes, eupts, pts)


class ControllerPlugin(BasePlugin):
    name = 'controller'
    systems = ['*']
    formulations = ['dual', 'std']

    def __init__(self, intg, cfgsect, suffix):
        super().__init__(intg, cfgsect, suffix)

        # Underlying elements class
        self.elementscls = intg.system.elementscls

        # Process frequency and other params
        self.nsteps = self.cfg.getint(cfgsect, 'nsteps')
        self.save_data = self.cfg.getint(cfgsect, 'savedata')
        self.set_omega = self.cfg.getint(cfgsect, 'setomega')
        self.perform_mpc = (self.cfg.getint(cfgsect, 'mpc') == 1)

        # List of points to be sampled and format
        self.pts = self.cfg.getliteral(cfgsect, 'samp-pts')
        self.fmt = self.cfg.get(cfgsect, 'format', 'primitive')

        # If performing mpc, then load network
        if self.perform_mpc:
            with open('/home/sisl/jeremy/deep_cfd/koopman/args.pkl', 'rb') as f:                                                                                                                                                            
                args = pickle.load(f)
            self.args =  args

            # Define array to hold old time snapshots and control inputs of the system
            self.X = np.zeros((int(args.seq_length/2) + 1, 128, 256, 4), dtype=np.float32)
            self.u = np.zeros((int(args.seq_length/2), args.action_dim), dtype=np.float32)

            # Define checkpoint name
            self.ckpt_name = self.cfg.get(cfgsect, 'checkpoint')

            # Define directory containing base flow solution
            base_dir = self.cfg.get(cfgsect, 'base_dir')

            # Set constraints for mpc
            self.R = self.cfg.getfloat(cfgsect, 'R')
            self.u_max = self.cfg.getfloat(cfgsect, 'u_max')

            # Initialize predicted state
            self.x_pred = np.zeros(args.code_dim)

            # Run script to find desired attributes and store them
            command = "python /home/sisl/jeremy/deep_cfd/koopman/find_matrices.py " + self.ckpt_name + " " + base_dir
            subprocess.call(command.split())

            # Load desired attributes from file
            f = h5py.File('./matrices_misc.h5', 'r')
            self.B = np.array(f['B'])
            self.shift_u = f['shift_u'].value[0]
            self.scale_u = f['scale_u'].value[0]
            self.goal_state = np.array(f['goal_state'])

        # Initial omega
        intg.system.omega = 0

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # MPI rank responsible for each point and rank-indexed info
        self._ptsrank = ptsrank = []
        self._ptsinfo = ptsinfo = [[] for i in range(comm.size)]

        # Physical location of the solution points
        plocs = [p.swapaxes(1, 2) for p in intg.system.ele_ploc_upts]

        # Load map from point to index                                                                                                                                                                     
        with open('loc_to_idx.pkl', 'rb') as f:                                                                                                                                                            
            self.loc_to_idx = pickle.load(f)

        # Locate the closest solution points in our partition
        closest = _closest_upts(intg.system.ele_types, plocs, self.pts)

        # Process these points
        for cp in closest:
            # Reduce over the distance
            _, mrank = comm.allreduce((cp[0], rank), op=get_mpi('minloc'))

            # Store the rank responsible along with its info
            ptsrank.append(mrank)
            ptsinfo[mrank].append(
                comm.bcast(cp[1:] if rank == mrank else None, root=mrank)
            )

    def _process_samples(self, samps):
        samps = np.array(samps)

        # If necessary then convert to primitive form
        if self.fmt == 'primitive' and samps.size:
            samps = self.elementscls.con_to_pri(samps.T, self.cfg)
            samps = np.array(samps).T

        return samps.tolist()

    # Find A-matrix and initial code value from neural network
    def _find_dynamics(self):
        # Save X and u to file
        f = h5py.File('./X_u.h5', 'w')
        f['X'] = self.X
        f['u'] = self.u
        f.close()

        # Run python script to find A matrix and initial state
        command = "python /home/sisl/jeremy/deep_cfd/koopman/find_dynamics.py " + self.ckpt_name
        subprocess.call(command.split())

        # Load desired values from file and return
        f = h5py.File('A_x0.h5', 'r')
        A = np.array(f['A'])
        x0 = np.array(f['x0'])

        return A, x0

    # Perform MPC optimization to find next input
    # Following example from CVXPY documentation
    def _find_mpc_input(self, A, B, x0):
        # First define prediction horizon
        T = 16

        # Define variables
        x = Variable(self.args.code_dim, T+1)
        u = Variable(self.args.action_dim, T)

        # Define costs for states and inputs
        Q = np.eye(self.args.code_dim)
        R = self.R*np.eye(self.args.action_dim)

        # Construct and solve optimization problem
        states = []
        for t in range(T):
            cost = quad_form((x[:,t+1] - self.goal_state), Q) + quad_form((u[:,t]*self.scale_u + self.shift_u), R)
            constr = [x[:,t+1] == A*x[:,t] + B*u[:,t],
                        norm(u[:,t]*self.scale_u + self.shift_u, 'inf') <= self.u_max]
            states.append(Problem(Minimize(cost), constr))
        
        # Sum problem objectives and concatenate constraints
        prob = sum(states)
        prob.constraints += [x[:,0] == x0]
        prob.solve()
        x1 = np.array([x.value[i, 1] for i in range(x.value.shape[0])])
        try:
            return u.value[0, 0]*self.scale_u + self.shift_u # Change if not scalar input
        except:
            return 0.0

    def __call__(self, intg):
        # Return if there is nothing to do for this step
        if (intg.nacptsteps % self.nsteps):
            return

        # MPI info
        comm, rank, root = get_comm_rank_root()

        # Solution matrices indexed by element type
        solns = dict(zip(intg.system.ele_types, intg.soln))

        # Points we're responsible for sampling
        ourpts = self._ptsinfo[comm.rank]

        # Sample the solution matrices at these points
        samples = [solns[et][ui, :, ei] for _, et, (ui, ei) in ourpts]
        samples = self._process_samples(samples)

        # Gather to the root rank to give a list of points per rank
        samples = comm.gather(samples, root=root)

        # If we're the root rank process the data
        if rank == root:
            data = []

            # Collate
            iters = [zip(pi, sp) for pi, sp in zip(self._ptsinfo, samples)]

            for mrank in self._ptsrank:
                # Unpack
                (ploc, etype, idx), samp = next(iters[mrank])

                # Determine the physical mesh rank
                prank = intg.rallocs.mprankmap[mrank]

                # Prepare the output row [[x, y], [rho, rhou, rhouv, E]]
                row = [ploc, samp]

                # Append
                data.append(row)

            # Define info for saving to file
            save_dir = '../sol_data/'
            list_of_files = glob.glob(save_dir + '*')
            latest_file = max(list_of_files, key=os.path.getctime)
            file_num = int(latest_file[-7:-3])

            # Save data in desired format
            # Define freestream values for to be used for cylinder
            rho = 1.0
            P = 1.0
            u = 0.236
            v = 0.0
            e = P/rho/0.4 + 0.5*(u**2 + v**2)
            freestream = np.array([rho, rho*u, rho*v, e])
            sol_data = np.zeros((128, 256, 4))
            sol_data[:, :] = freestream
            for i in range(len(self.loc_to_idx.keys())):
                idx1, idx2 = self.loc_to_idx[i]
                sol_data[idx1, idx2] = data[i][1]

            # Update running total of previous states
            if self.perform_mpc: self.X = np.vstack((self.X[1:], np.expand_dims(sol_data, axis=0)))

            # Initialize values
            t = intg.tcurr
            self.t_old = t
            pred_error = 0.0

            if self.set_omega == 0:
                omega = 0.0    
            elif self.perform_mpc:
                # Find model of system and determine optimal input with MPC
                try:
                    A, x0 = self._find_dynamics()
                    if np.linalg.norm(self.X[0]) > 0.0:
                        u0 = (self.u[-1] - self.shift_u)/self.scale_u
                        omega = self._find_mpc_input(A, self.B, x0)
                    else:
                        omega = 0.0 # No input if insufficient data to construct dynamical model
                except:
                    omega = 0.0
                self.u = np.concatenate((self.u[1:], np.expand_dims(np.array([omega]), axis=0)))
            else:
                # To generate training data
                # Have no inputs for periods, otherwise sinusoidal
                if t % 1000 > 900:
                    omega = 0.0
                else:
                    freq = 2*np.pi * (t/1000)/500.0
                    omega = 0.3*np.sin(freq*t)


                # Proportional control
                # location = 88 # re50 
                # gain = 0.4 # re50
                # rho = sol_data[64, location, 0]
                # rho_v = sol_data[64, location, 2]
                # omega = gain*rho_v/rho
                

            # Save data if desired
            if self.save_data == 1:
                # Save to h5 file
                file_num += 1
                filename = save_dir + 'sol_data_' + str(file_num).zfill(4) + '.h5'
                f = h5py.File(filename, 'w')
                f['sol_data'] = sol_data
                f['control_input'] = omega
                if self.perform_mpc: f['cost'] = np.linalg.norm(self.goal_state - x0)
                f.close()
        else:
            omega = None

        # Broadcast omega to all of the MPI ranks
        intg.system.omega = float(comm.bcast(omega, root=root))
