import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataloader import DataLoader
import h5py
import tensorflow as tf
import numpy as np
import time
from bayes_filter import BayesFilter
from utils import reconstruct_bayes_filter
import random

def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',          type= str,  default='bf_model', help='name of checkpoint files for saving')

    parser.add_argument('--seq_length',         type=int,   default= 16,        help='sequence length for training')
    parser.add_argument('--batch_size',         type=int,   default= 2,         help='minibatch size')
    parser.add_argument('--code_dim',           type=int,   default= 32,        help='dimensionality of code')
    parser.add_argument('--noise_dim',          type=int,   default= 32,        help='dimensionality of noise vector')

    parser.add_argument('--num_epochs',         type=int,   default= 50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.00075,   help='learning rate')
    parser.add_argument('--decay_rate',         type=float, default= 0.75,      help='decay rate for learning rate')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')

    parser.add_argument('--data_dir',           type=str,   default='',         help='directory containing cylinder data')
    parser.add_argument('--n_sequences',        type=int,   default= 1,         help='number of files to load for training')
    parser.add_argument('--min_num',            type=int,   default= 0,         help='lowest number time snapshot to load for training')
    parser.add_argument('--max_num',            type=int,   default= 5000,      help='highest number time snapshot to load for training')
    parser.add_argument('--stagger',            type=int,   default= 500,       help='number of time steps between training examples')
    parser.add_argument('--control_input',      type=bool,  default= False,     help='whether to account for control input in modeling dynamics')
    parser.add_argument('--rec_vel',            type=bool,  default= False,     help='whether to reconstruct velocity values from codes')

    ######################################
    #    Feature Extractor Params        #
    ######################################
    parser.add_argument('--num_filters', nargs='+', type=int, default=[32],     help='number of filters after each down/uppconv')
    parser.add_argument('--reg_weight',         type=float, default= 1e-4,      help='weight applied to regularization losses')
    parser.add_argument('--kl_weight',          type=float, default= 0.0,       help='weight applied to kl-divergence loss, annealed if zero')
    parser.add_argument('--feature_dim',        type=int,   default= 32,        help='dimensionality of extracted features')

    ######################################
    #          Additional Params         #
    ######################################
    parser.add_argument('--rnn_size',           type=int, default=64,           help='size of rnn layer')
    parser.add_argument('--transform_size',     type=int, default=64,           help='generic size of layers performing transformations')
    parser.add_argument('--num_matrices',       type=int, default=4,            help='number of matrices to be combined for propagation')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[32],  help='size of inference network')

    ######################################
    #      Reconstruct Solution?         #
    ######################################
    parser.add_argument('--reconstruct',        type=bool,  default=False,      help='whether to load saved model for reconstructed solution')
    parser.add_argument('--generative',         type=bool,  default=False,      help='whether to generate solutions (instead of reconstructing)')
    parser.add_argument('--solution_number',    type=int,   default= 1,         help='which solution to reconstruct')
    parser.add_argument('--n_seq',              type=int,   default= 1,         help='number of sequences to propagate')
    parser.add_argument('--key_prefix',         type=str,   default='rec_koopman', help='prefix for key in h5 file')

    args = parser.parse_args()

    # Set random seed
    random.seed(1)

    # Construct model
    net = BayesFilter(args)

    # Either reconstruct or train
    if args.reconstruct:
        reconstruct_bayes_filter(args, net)
    else:
        train(args, net)

# Train network
def train(args, net):
    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Load data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        data_loader = DataLoader(args, shift, scale)
        sess.run(tf.assign(net.shift, data_loader.shift_x))
        sess.run(tf.assign(net.scale, data_loader.scale_x))

        #Function to evaluate loss on validation set
        def val_loss():
            data_loader.reset_batchptr_val()
            loss = 0.0
            for b in xrange(data_loader.n_batches_val):
                # Get inputs
                batch_dict = data_loader.next_batch_val()
                x = batch_dict["inputs"]

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (args.batch_size*(args.seq_length+1), 128, 256, 4))
                if args.kl_weight > 0.0:
                    feed_in[net.kl_weight] = args.kl_weight
                else:
                    feed_in[net.kl_weight] = 1.0

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/data_loader.n_batches_val

        # Initialize variable to track validation score over time
        old_score = 1e9
        count_decay = 0
        decay_epochs = []

        # Set initial learning rate and weight on kl divergence
        print 'setting learning rate to ', args.learning_rate
        sess.run(tf.assign(net.learning_rate, args.learning_rate))

        # Define temperature for annealing kl_weight
        T = 5*data_loader.n_batches_train
        count = 0

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Initialize loss
            loss = 0.0
            rec_loss = 0.0
            kl_loss = 0.0

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if (old_score - score) < 0.01:
                count_decay += 1
                decay_epochs.append(e)
                if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                print 'setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay)
                sess.run(tf.assign(net.learning_rate, args.learning_rate * (args.decay_rate ** count_decay)))
            old_score = score
            data_loader.reset_batchptr_train()

            print 'learning rate is set to ', args.learning_rate * (args.decay_rate ** count_decay)

            # Loop over batches
            for b in xrange(data_loader.n_batches_train):
                start = time.time()
                count += 1

                # Update kl_weight
                if args.kl_weight > 0.0:
                    kl_weight = args.kl_weight
                else:
                    kl_weight = min(0.1, 0.01 + count/float(T))

                # Get inputs
                batch_dict = data_loader.next_batch_train()
                x = batch_dict["inputs"]

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (args.batch_size*(args.seq_length+1), 128, 256, 4))
                feed_in[net.kl_weight] = kl_weight

                # Find loss and perform training operation
                feed_out = [net.cost, net.loss_reconstruction, net.kl_loss, net.train]
                out = sess.run(feed_out, feed_in)

                # Update and display cumulative losses
                loss += out[0]
                rec_loss += out[1]
                kl_loss += out[2]

                end = time.time()

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, loss/10., end - start)
                    print "{}/{} (epoch {}), rec_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, rec_loss/10., end - start)
                    print "{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, kl_loss/10., end - start)
                    print ''
                    loss = 0.0
                    rec_loss = 0.0
                    kl_loss = 0.0

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print "model saved to {}".format(checkpoint_path)

if __name__ == '__main__':
    main()
