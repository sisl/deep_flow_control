'''
This script is used to load a saved checkpoint in order to find
the B-matrix, action normalization parameters, and goal state
encoding. This script is called prior to performing flow control,
as they are required to perform MPC.

Note that all paths need to be specified correctly in the config file.
'''

import os, sys
import h5py
import tensorflow as tf
import numpy as np
import json
import argparse

from koopman_model import KoopmanModel

# Read in args
with open(sys.argv[1] + '/args.json') as args_dict:
    args_dict = json.load(args_dict,)
args = argparse.Namespace()
for (k, v) in args_dict.items():
    vars(args)[k] = v
# Construct model
net = KoopmanModel(args)

# Begin session and assign parameter values
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    ckpt_name = sys.argv[2]
    saver.restore(sess, sys.argv[1] + '/' + args.save_dir + '/' + ckpt_name)

    # Find B-matrix (will be constant)
    B = sess.run(net.B)[0].T

    # To find goal state, find encoding of steady base flow at Re50
    x = np.zeros((args.batch_size*(args.seq_length+1), 128, 256, 4), dtype=np.float32)
    u = np.zeros((args.batch_size, args.seq_length, args.action_dim), dtype=np.float32)

    # Load solution for base flow
    f = h5py.File(sys.argv[3], 'r')
    x[0] = np.array(f['sol_data'])

    # Normalize data
    x = (x - sess.run(net.shift))/sess.run(net.scale)

    # Run inputs through network, find encoding
    feed_in = {}
    feed_in[net.x] = x
    feed_in[net.u] = u
    feed_out = net.code_x
    code_x = sess.run(feed_out, feed_in)

    # Define goal state
    goal_state = code_x[0]

    # Save quantities to file
    f = h5py.File('./matrices_misc.h5', 'w')
    f['B'] = B
    f['goal_state'] = goal_state
    f.close()
