import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
from dataloader import DataLoader
import h5py
import json
import tensorflow as tf
import numpy as np
import time
from koopman_model import KoopmanModel
from utils import reconstruct_koopman_2d
import random

def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',           type=str,   default='./checkpoints', help='directory to store checkpointed models')
    parser.add_argument('--val_frac',           type=float, default=0.1,        help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name',          type= str,  default='',         help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name',          type= str,  default='koopman_model', help='name of checkpoint files for saving')

    parser.add_argument('--seq_length',         type=int,   default= 32,        help='sequence length for training')
    parser.add_argument('--batch_size',         type=int,   default= 1,         help='minibatch size')
    parser.add_argument('--code_dim',           type=int,   default= 64,        help='dimensionality of code')
    parser.add_argument('--action_dim',         type=int,   default= 1,         help='actions dimensionality')

    parser.add_argument('--num_epochs',         type=int,   default= 50,        help='number of epochs')
    parser.add_argument('--learning_rate',      type=float, default= 0.00075,   help='learning rate')
    parser.add_argument('--decay_rate',         type=float, default= 0.5,       help='decay rate for learning rate')
    parser.add_argument('--l2_regularizer',     type=float, default= 10.0,      help='regularization for least squares')
    parser.add_argument('--grad_clip',          type=float, default= 5.0,       help='clip gradients at this value')

    parser.add_argument('--data_dir',           type=str,   default='',         help='directory containing cylinder data')
    parser.add_argument('--n_sequences',        type=int,   default= 1200,      help='number of files to load for training')
    parser.add_argument('--min_num',            type=int,   default= 0,         help='lowest number time snapshot to load for training')
    parser.add_argument('--max_num',            type=int,   default= 5000,      help='highest number time snapshot to load for training')
    parser.add_argument('--recursive_pred',     type=bool,  default= True,      help='whether to generate recursive predictions for y')
    parser.add_argument('--halve_seq',          type=bool,  default= True,      help='whether to generate A-matrix based on only half of sequence')
    parser.add_argument('--control_input',      type=bool,  default= False,     help='whether to account for control input in modeling dynamics')
    parser.add_argument('--start_file',         type=int,   default= 100,       help='first file number to load for training')
    parser.add_argument('--stagger',            type=int,   default= 1,         help='number of time steps between training examples')

    ######################################
    #          Network Params            #
    ######################################
    parser.add_argument('--num_filters', nargs='+', type=int, default=[32],     help='number of filters after each down/uppconv')
    parser.add_argument('--reg_weight',         type=float, default= 1e-4,      help='weight applied to regularization losses')
    parser.add_argument('--code_weight',         type=float, default= 0.0,      help='weight applied to code loss')

    ######################################
    #      Reconstruct Solution?         #
    ######################################
    parser.add_argument('--reconstruct',        type=bool,  default=False,      help='whether to load saved model for reconstructed solution')
    parser.add_argument('--save_codes',         type=bool,  default=False,      help='whether to save all solutions to code values')
    parser.add_argument('--solution_number',    type=int,   default= 0,         help='which solution to reconstruct')
    parser.add_argument('--n_seq',              type=int,   default= 1,         help='number of sequences to propagate')
    parser.add_argument('--key_prefix',         type=str,   default='rec_koopman', help='prefix for key in h5 file')

    args = parser.parse_args()

    # Set random seed
    random.seed(1)

    # Construct model
    net = KoopmanModel(args)

    # Either reconstruct or train
    if args.reconstruct:
        reconstruct_koopman_2d(args, net)
    else:
        # Only dump args if training
        with open('args.json', 'w') as f:
            json.dump(vars(args), f)
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

        # Find normalization params if stored
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)

        # Load data and store normalization params
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
                if args.control_input: u = batch_dict["actions"]

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (args.batch_size*(args.seq_length+1), 128, 256, 4))
                if args.control_input: feed_in[net.u] = u[:, :args.seq_length]

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

        # Evaluate loss on validation set
        score = val_loss()
        print('Validation Loss: {0:f}'.format(score))

        # Loop over epochs
        for e in xrange(args.num_epochs):

            # Initialize loss
            loss = 0.0
            rec_loss = 0.0

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

                # Get inputs
                batch_dict = data_loader.next_batch_train()
                x = batch_dict["inputs"]
                if args.control_input: u = batch_dict["actions"]

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (args.batch_size*(args.seq_length+1), 128, 256, 4))
                if args.control_input: feed_in[net.u] = u[:, :args.seq_length]

                # Find loss and perform training operation
                feed_out = [net.cost, net.loss_reconstruction, net.train]
                
                # Sometimes things blow up when it tries to take least-squares
                try:
                    out = sess.run(feed_out, feed_in)
                except:
                    out = [0.0, 0.0, 0.0]

                # Update and display cumulative losses
                loss += out[0]
                rec_loss += out[1]

                end = time.time()

                # Print loss
                if (e * data_loader.n_batches_train + b) % 10 == 0 and b > 0:
                    print "{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, loss/10, end - start)
                    print "{}/{} (epoch {}), rec_loss = {:.3f}, time/batch = {:.3f}" \
                      .format(e * data_loader.n_batches_train + b, args.num_epochs * data_loader.n_batches_train,
                              e, rec_loss/10, end - start)
                    print ''
                    loss = 0.0
                    rec_loss = 0.0

            # Save model every epoch
            checkpoint_path = os.path.join(args.save_dir, args.save_name + '.ckpt')
            saver.save(sess, checkpoint_path, global_step = e)
            print "model saved to {}".format(checkpoint_path)

            # Evaluate loss on validation set
            score = val_loss()
            print('Validation Loss: {0:f}'.format(score))

if __name__ == '__main__':
    main()
