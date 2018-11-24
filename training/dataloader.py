import h5py
import math
import numpy as np
import random
import progressbar

# Class to load and preprocess data
class DataLoader():
    def __init__(self, args, shift, scale, net=None, sess=None):
        self.batch_size = args.batch_size*args.seq_length
        self.shift_x = shift
        self.scale_x = scale
        self.net = net
        self.sess = sess
        self.seq_length = args.seq_length
        self.control_input = args.control_input

        print 'validation fraction: ', args.val_frac

        print "loading data..."
        self._load_data_2d(args)
        self._create_inputs_targets(args)

        print 'creating splits...'
        self._create_split(args)

        print 'shifting/scaling data...'
        self._shift_scale(args)

    def _load_data_2d(self, args):
        n_files = args.n_sequences*(args.seq_length+1)
        max_gap = (args.max_num - args.stagger*args.seq_length)/args.n_sequences
        start_idxs = np.linspace(args.min_num, max_gap*args.n_sequences, args.n_sequences)
        file_nums = np.array([])
        for i in xrange(args.n_sequences):
            file_nums = np.concatenate([file_nums, np.linspace(start_idxs[i], start_idxs[i] + args.seq_length*args.stagger, args.seq_length+1)])

        # Define progress bar
        bar = progressbar.ProgressBar(maxval=n_files).start()

        # Load data
        # Initialize x
        x = np.zeros((n_files, 128, 256, 4), dtype=np.float32)
        if args.control_input:
            u = np.zeros((n_files, args.action_dim), dtype=np.float32)

        for i in xrange(n_files):
            f = h5py.File(args.data_dir + 'sol_data_'+str(int(file_nums[i])).zfill(4)+'.h5', 'r')
            x[i] = np.array(f['sol_data'])
            if args.control_input: u[i] = np.array(f['control_input'])
            bar.update(i)

        # Divide into sequences
        self.x = x.reshape(-1, args.seq_length+1, 128, 256, 4)
        self.x = self.x[:int(np.floor(len(self.x)/args.batch_size)*args.batch_size)]
        self.x = self.x.reshape(-1, args.batch_size, args.seq_length+1, 128, 256, 4)

        if args.control_input:
            self.u = u.reshape(-1, args.seq_length+1, args.action_dim)
            self.u = self.u[:int(np.floor(len(self.u)/args.batch_size)*args.batch_size)]
            self.u = self.u.reshape(-1, args.batch_size, args.seq_length+1, args.action_dim)

    def _create_inputs_targets(self, args):
        # Create batch_dict and permuatation
        self.batch_dict = {}
        p = np.random.permutation(len(self.x))

        # Print tensor shapes
        print 'inputs: ', self.x.shape
        self.batch_dict["inputs"] = np.zeros((args.batch_size, args.seq_length+1, 128, 256, 4))
        if args.control_input: self.batch_dict["actions"] = np.zeros((args.batch_size, args.seq_length+1, args.action_dim))

        # Shuffle data
        print 'shuffling...'
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        if args.control_input: self.u = self.u[p]

    # Separate data into train/validation sets
    def _create_split(self, args):
        # compute number of batches
        self.n_batches = len(self.x)
        self.n_batches_val = int(math.floor(args.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print 'num training batches: ', self.n_batches_train
        print 'num validation batches: ', self.n_batches_val

        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Shift and scale data to be zero-mean, unit variance
    def _shift_scale(self, args):
        # Find means and std if not initialized to anything
        if np.sum(self.scale_x) == 0.0:
            self.shift_x = np.mean(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4))
            self.scale_x = np.std(self.x[:self.n_batches_train], axis=(0, 1, 2, 3, 4))

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[self.batchptr_train]
        self.batch_dict["inputs"] = (self.x[batch_index] - self.shift_x)/self.scale_x
        if self.control_input: self.batch_dict["actions"] = self.u[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(self.n_batches_train)
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = self.batchptr_val + self.n_batches_train-1
        self.batch_dict["inputs"] = (self.x[batch_index] - self.shift_x)/self.scale_x
        if self.control_input: self.batch_dict["actions"] = self.u[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return next batch of data in validation set
    def random_batch_val(self):
        # Extract next validation batch
        batch_index = random.randint(self.n_batches_train, self.n_batches-1)
        self.batch_dict["inputs"] = (self.x[batch_index] - self.shift_x)/self.scale_x
        if self.control_input: self.batch_dict["actions"] = self.u[batch_index]
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

