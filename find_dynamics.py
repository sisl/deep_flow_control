'''
This script is used to load a saved checkpoint in order to find
the A-matrix, and initial state encoding based on a previous 
sequence of observed states and actions.

Note that all paths are specific to the machine on which the 
code was run and need to be updated accordingly.
'''

import os, sys
import h5py
import tensorflow as tf
import numpy as np
import pickle

from koopman_model import KoopmanModel

# Read in args that are stored when training network 
# Path must be changed depending on where this is stored
with open('/home/sisl/jeremy/deep_cfd/koopman/args.pkl', 'rb') as f:                                                                                                                                                            
    args = pickle.load(f)

# Construct model
net = KoopmanModel(args)

# Begin session and assign parameter values
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    ckpt_name = sys.argv[1]

    # Load model from checkpoint (again, path must be changed)
    saver.restore(sess, '/home/sisl/jeremy/deep_cfd/koopman/' + args.save_dir + '/' + ckpt_name)

    # Read in state data
    f = h5py.File('./X_u.h5', 'r')
    X = np.array(f['X'])
    u = np.array(f['u']) 
    f.close()

    # Construct input to network (second half of data will not matter)
    x = np.vstack((X, X[:args.seq_length/2]))
    x = (x - sess.run(net.shift))/sess.run(net.scale)
    u = (u - sess.run(net.shift_u))/sess.run(net.scale_u)

    # Format control inputs
    controls = np.concatenate((u, u))
    controls = np.expand_dims(controls, axis=0)

    # Run inputs through network, find dynamics and initial state mapping
    feed_in = {}
    feed_in[net.x] = x
    feed_in[net.u] = controls
    feed_out = [net.A, net.code_x]
    A, code_x = sess.run(feed_out, feed_in)
    code_x = code_x.reshape(args.batch_size, args.seq_length, args.code_dim)

    # Extract desired info and save to file
    A = A[0].T
    x0 = code_x[0, args.seq_length/2]
    f = h5py.File('./A_x0.h5', 'w')
    f['A'] = A
    f['x0'] = x0
    f.close()    


