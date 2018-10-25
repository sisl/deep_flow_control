import os, sys
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import numpy as np
import random
import progressbar
from bayes_filter import BayesFilter

# Reconstruct koopman solution for 2d problem (just save relevant quantities in an h5 file)
def reconstruct_koopman_2d(args, net):
    # Begin tf session
    with tf.Session() as sess: 
        # Initialize variables
        tf.global_variables_initializer().run()
        restore_vars = [v for v in tf.global_variables() if 'Adam' not in v.name]
        saver = tf.train.Saver(restore_vars, max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Define numbers of files to load and begin loading them
        file_nums = np.linspace(args.solution_number, args.solution_number+args.stagger*args.n_seq*args.seq_length, args.n_seq*args.seq_length+1)

        # Load data
        # Initialize x (and u if inputs)
        x = np.zeros((args.n_seq*args.seq_length+1, 128, 256, 4), dtype=np.float32)
        if args.control_input:
            u = np.zeros((args.n_seq*args.seq_length+1, args.action_dim), dtype=np.float32)
        for i in xrange(args.n_seq*args.seq_length+1):
            f = h5py.File(args.data_dir + 'sol_data_'+str(int(file_nums[i])).zfill(4)+'.h5', 'r')
            x[i] = np.array(f['sol_data'])
            if args.control_input: u[i] = np.array(f['control_input'])

        # Store original data
        orig_data = x

        # Normalize x
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        x = (x - shift)/scale

        # Initialize array to hold reconstructed solution
        pred_sol = np.zeros_like(x, dtype=np.float32)
        codes = np.zeros((args.seq_length, args.code_dim), dtype=np.float32)

        # Extract data for generating A-matrix
        x = x[:(args.seq_length+1)]

        # Get inputs and targets (staggered by 1 time step)
        x = np.tile(x, (args.batch_size, 1, 1, 1))
        if args.control_input: u = np.tile(np.expand_dims(u, axis=0), (args.batch_size, 1, 1))

        # Find A matrix and reconstructed solution for first time step first
        print 'generating predicted solutions...'
        feed_in = {}
        feed_in[net.x] = x
        if args.control_input: 
            feed_in[net.u] = u[:, :args.seq_length]
            feed_out = [net.A, net.B, net.code_x, net.code_y_reshape, net.rec_sol]
            A, B, code_x, code_y, rec_sol = sess.run(feed_out, feed_in)
            B = B[0]
        else:
            feed_out = [net.A, net.code_x, net.code_y_reshape, net.rec_sol]
            A, code_x, code_y, rec_sol = sess.run(feed_out, feed_in)
        A = A[0]
        code_x = code_x[0]

        # Fill in first element of predicted solutions
        pred_sol[0] = rec_sol[0]

        # Generate code predictions
        for i in xrange(args.n_seq):
            for t in xrange(args.seq_length):
                if t == 0:
                    codes[t] = np.dot(A.T, code_x)
                else:
                    codes[t] = np.dot(A.T, codes[t-1])
                if args.control_input:
                    codes[t] += np.dot(B.T, u[0, i*args.seq_length + t])

            # Now find reconstruction of codes
            feed_in = {}
            feed_in[net.code] = np.tile(codes, (2*args.batch_size, 1))
            feed_out = net.rec_sol
            rec_sol = sess.run(feed_out, feed_in)
            pred_sol[i*args.seq_length+1:(i+1)*args.seq_length+1] = rec_sol[:args.seq_length]
            code_x = codes[-1]

        # Save original solution, predicted solution, and A matrix to h5 file
        print 'saving to file...'
        f = h5py.File(args.key_prefix + '.h5', 'w')
        f['x'] = orig_data
        f['pred_sol'] = pred_sol*scale + shift
        f['A'] = A
        if args.control_input:
            f['u'] = u
            f['B'] = B
        f.close()
        print 'done.'

# Reconstruct koopman solution for 2d problem (just save relevant quantities in an h5 file)
def reconstruct_bayes_filter(args, net):
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        restore_vars = [v for v in tf.global_variables() if 'generate_flag' not in v.name]
        saver = tf.train.Saver(restore_vars, max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Define numbers of files to load and begin loading them
        file_nums = np.linspace(args.solution_number, args.solution_number+args.stagger*args.n_seq*args.seq_length, args.n_seq*args.seq_length+1)

        # Initialize x (and u if inputs)
        x = np.zeros((args.n_seq*args.seq_length+1, 128, 256, 4), dtype=np.float32)
        for i in xrange(args.n_seq*args.seq_length+1):
            f = h5py.File(args.data_dir + 'sol_data_'+str(int(file_nums[i])).zfill(4)+'.h5', 'r')
            x[i] = np.array(f['sol_data'])

        # Store original data
        orig_data = x

        # Normalize x
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        x = (x - shift)/scale

        # Initialize array to hold reconstructed solution
        pred_sol = np.zeros_like(x, dtype=np.float32)
        codes = np.zeros((args.seq_length, args.code_dim), dtype=np.float32)

        # Extract data for generating A-matrix
        x = x[:(args.seq_length+1)]

        # Get inputs and targets (staggered by 1 time step)
        x = np.tile(x, (args.batch_size, 1, 1, 1))

        # Find reconstructed solution
        print 'generating predicted solutions...'
        feed_in = {}
        feed_in[net.x] = x
        feed_out = [net.rec_sol, net.z_pred]
        rec_sol, z_pred = sess.run(feed_out, feed_in)
        pred_sol[:(args.seq_length+1)] = rec_sol[:(args.seq_length+1)]

        # Generate additional predictions if desired
        for i in xrange(1, args.n_seq):
            z_pred = z_pred.reshape(args.batch_size, (args.seq_length+1), args.code_dim)
            z1 = z_pred[:, -1]
            feed_in = {}
            feed_in[net.z1] = z1
            feed_in[net.generative] = True
            feed_out = [net.rec_sol, net.z_pred]
            rec_sol, z_pred = sess.run(feed_out, feed_in)
            pred_sol[i*args.seq_length+1:(i+1)*args.seq_length+1] = rec_sol[:args.seq_length]

        # Save original solution, predicted solution, and A matrix to h5 file
        print 'saving to file...'
        f = h5py.File(args.key_prefix + '.h5', 'w')
        f['pred_sol'] = pred_sol*scale + shift
        f.close()
        print 'done.'

