import numpy as np
import tensorflow as tf

class BayesFilter():
    def __init__(self, args):

        # Placeholder for data -- inputs are number of elements x pts in mesh x dimensionality of data for each point
        self.x = tf.Variable(np.zeros((args.batch_size*(args.seq_length+1), 128, 256, 4), dtype=np.float32), trainable=False, name="input_data")

        # Parameters to be set externally
        self.is_training = tf.Variable(False, trainable=False, name="training_flag")
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")
        self.kl_weight = tf.Variable(0.0, trainable=False, name="kl_weight")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(4), trainable=False, name="input_shift")
        self.scale = tf.Variable(np.zeros(4), trainable=False, name="input_scale")
        self.generative = tf.Variable(args.generative, trainable=False, name="generate_flag")
        
        # Create the computational graph
        self._create_feature_extractor(args)
        self._create_initial_generator(args)
        self._create_transition_matrices(args)
        self._create_weight_network_params(args)
        self._create_inference_network_params(args)
        self._propagate_solution(args)
        self._create_decoder(args)
        self._create_optimizer(args)

    # Define conv operation depending on number of dimensions in input
    def _conv_operation(self, in_tensor, num_filters, kernel_size, args, name, transpose=False, stride=1):
        if transpose:
            return tf.layers.conv2d_transpose(in_tensor, 
                                num_filters, 
                                kernel_size=kernel_size,
                                strides=stride, 
                                padding='same', 
                                name=name, 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        else:
            return tf.layers.conv2d(in_tensor, 
                                    num_filters, 
                                    kernel_size=kernel_size,
                                    strides=stride, 
                                    padding='same', 
                                    name=name, 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Code for initalizing resnet layer in encoder
    # Based on https://arxiv.org/pdf/1603.05027.pdf 
    # order is BN -> activation -> weights -> BN -> activation -> weights
    def _create_bottleneck_layer(self, in_tensor, args, name, num_filters):

        bn_output1 = tf.layers.batch_normalization(in_tensor, training=self.is_training)
        act_output1 = tf.nn.relu(bn_output1)
        conv_output1 = self._conv_operation(act_output1, num_filters/2, 1, args, name+'_conv1')

        bn_output2 = tf.layers.batch_normalization(conv_output1, training=self.is_training)
        act_output2 = tf.nn.relu(bn_output2)
        conv_output2 = self._conv_operation(act_output2, num_filters/2, 3, args, name+'_conv2')

        bn_output3 = tf.layers.batch_normalization(conv_output2, training=self.is_training)
        act_output3 = tf.nn.relu(bn_output3)
        bottleneck_output = self._conv_operation(act_output3, num_filters, 1, args, name+'_conv3')

        return bottleneck_output

    # Encoding Stage
    # Pattern: downconv with stride 2 followed by resnet bottleneck layer
    def _create_feature_extractor(self, args):
        # Series of downconvolutions and bottleneck layers
        downconv_input = self.x

        for i in xrange(len(args.num_filters)):
            downconv_output = self._conv_operation(downconv_input, args.num_filters[i], 3, args, 'downconv'+str(i), stride=2)
            bottleneck_output = self._create_bottleneck_layer(downconv_output, args, 'bn'+str(i), args.num_filters[i])
            downconv_input = downconv_output + bottleneck_output
        self.encoder_conv_output = tf.nn.relu(downconv_input)

        # Fully connected layer to get code
        self.reshape_output = tf.reshape(self.encoder_conv_output, [args.batch_size*(args.seq_length+1), -1])
        code = tf.layers.dense(self.reshape_output, 
                                    args.feature_dim,
                                    name='to_features', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        
        # Reshape output
        self.features = tf.reshape(code, [args.batch_size, args.seq_length+1, args.feature_dim])

    # Function to generate samples given distribution parameters
    def _gen_sample(self, args, dist_params):
        w_mean, w_logstd = tf.split(dist_params, [args.noise_dim, args.noise_dim], axis=1)
        w_std = tf.exp(w_logstd) + 1e-3
        samples = tf.random_normal([args.batch_size, args.noise_dim])
        w = samples*w_std + w_mean
        w = tf.minimum(tf.maximum(w, -10.0), 10.0)
        w = tf.cond(self.generative, lambda: samples, lambda: w)  # Just sample from prior for generative model
        return w

    # Bidirectional LSTM to generate initial sample of w1, then form z1 from w1
    def _create_initial_generator(self, args):
        fwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        bwd_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer())
        
        # Get outputs from rnn and concatenate
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fwd_cell, bwd_cell, self.features, dtype=tf.float32)
        output_fw, output_bw = outputs
        output = tf.concat([output_fw[:, -1], output_bw[:, -1]], axis=1)

        # Single affine transformation into w1 distribution params
        hidden = tf.layers.dense(output, 
                                args.transform_size, 
                                activation=tf.nn.relu,
                                name='to_hidden_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.w1_dist = tf.layers.dense(hidden, 
                                2*args.noise_dim, 
                                name='to_w1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.w1 = self._gen_sample(args, self.w1_dist)

        # Now construct z1 through transformation with single hidden layer
        hidden = tf.layers.dense(self.w1, 
                                args.transform_size, 
                                activation=tf.nn.relu,
                                name='to_hidden_z1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.z1 = tf.layers.dense(hidden, 
                                args.code_dim, 
                                name='to_z1', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Initialize potential transition matrices
    def _create_transition_matrices(self, args):
        self.A_matrices = tf.get_variable("A_matrices", [args.num_matrices, args.code_dim, args.code_dim], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        self.C_matrices = tf.get_variable("C_matrices", [args.num_matrices, args.noise_dim, args.code_dim], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

    # Create parameters to comprise weight network
    def _create_weight_network_params(self, args):
        self.weight_w = []
        self.weight_b = []

        # Have single hidden layer and fully connected to output
        self.weight_w.append(tf.get_variable("weight_w1", [args.code_dim, args.transform_size], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.weight_w.append(tf.get_variable("weight_w2", [args.transform_size, args.num_matrices], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))

        self.weight_b.append(tf.get_variable("weight_b1", [args.transform_size], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.weight_b.append(tf.get_variable("weight_b2", [args.num_matrices], 
                                            regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))

    # Create parameters to comprise inference network
    def _create_inference_network_params(self, args):
        self.inference_w = []
        self.inference_b = []

        # Loop through elements of inference network and define parameters
        for i in xrange(len(args.inference_size)):
            if i == 0:
                prev_size = args.feature_dim+args.code_dim
            else:
                prev_size = args.inference_size[i-1]
            self.inference_w.append(tf.get_variable("inference_w"+str(i), [prev_size, args.inference_size[i]], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
            self.inference_b.append(tf.get_variable("inference_b"+str(i), [args.inference_size[i]]))

        # Last set of weights to map to output
        self.inference_w.append(tf.get_variable("inference_w_end", [args.inference_size[-1], 2*args.noise_dim], 
                                                regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight)))
        self.inference_b.append(tf.get_variable("inference_b_end", [2*args.noise_dim]))

    # Function to get weights for transition matrices
    def _get_weights(self, z):
        hidden = tf.nn.relu(tf.nn.xw_plus_b(z, self.weight_w[0], self.weight_b[0]))
        return tf.nn.softmax(tf.nn.xw_plus_b(hidden, self.weight_w[1], self.weight_b[1]))

    # Function to generate w sample from inference network
    def _get_inference_sample(self, args, features, z):
        inference_input = tf.concat([features, z], axis=1)
        for i in xrange(len(args.inference_size)):
            inference_input = tf.nn.relu(tf.nn.xw_plus_b(inference_input, self.inference_w[i], self.inference_b[i]))
        w_dist = tf.nn.xw_plus_b(inference_input, self.inference_w[-1], self.inference_b[-1])

        # Generate sample
        w = self._gen_sample(args, w_dist)
        return w_dist, w

    # Now use various params/networks to propagate solution forward in time
    def _propagate_solution(self, args):
        # Find current observation (expand dimension for stacking later)
        z_t = tf.expand_dims(self.z1, axis=1)

        # Initialize array for stacking observations and distribution params
        self.z_pred = [z_t]
        self.w_dists = [tf.expand_dims(self.w1_dist, axis=1)]

        # Loop through time and advance observation, get distribution params
        for t in xrange(args.seq_length):
            # Find A and C matrices
            weights = self._get_weights(tf.squeeze(z_t, axis=1))
            A_t_list = []
            C_t_list = []
            for i in xrange(args.batch_size):
                A_t_list.append(tf.add_n([weights[i, j]*self.A_matrices[j] for j in xrange(args.num_matrices)])) 
                C_t_list.append(tf.add_n([weights[i, j]*self.C_matrices[j] for j in xrange(args.num_matrices)]))
            A_t = tf.stack(A_t_list) 
            C_t = tf.stack(C_t_list) 

            # Draw noise sample and append sample to list
            w_dist, w_t = self._get_inference_sample(args, self.features[:, t+1], tf.squeeze(z_t, axis=1))
            self.w_dists.append(tf.expand_dims(w_dist, axis=1))

            # Now advance observation forward in time
            z_t = tf.matmul(z_t, A_t) + tf.matmul(tf.expand_dims(w_t, axis=1), C_t)
            self.z_pred.append(z_t)

        # Stack predictions into single tensor for reconstruction
        self.z_pred = tf.reshape(tf.stack(self.z_pred, axis=1), [args.batch_size*(args.seq_length+1), args.code_dim])
        self.w_dists = tf.reshape(tf.stack(self.w_dists, axis=1), [args.batch_size*(args.seq_length+1), 2*args.noise_dim])
        
    # Decoding stage -- reverse decoder to reconstruct input
    def _create_decoder(self, args):
        # Map back to dimensionality of features
        features_rec = tf.layers.dense(self.z_pred, 
                                args.feature_dim, 
                                name='to_features_rec', 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))

        # Reverse fully connected layer and reshape
        self.rev_fc_output = tf.layers.dense(features_rec, 
                                            self.reshape_output.get_shape().as_list()[-1], 
                                            activation=tf.nn.relu,
                                            name='from_code', 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        conv_shape = self.encoder_conv_output.get_shape().as_list()
        upconv_output = tf.reshape(self.rev_fc_output, conv_shape)

        # Series of bottleneck layers and upconvolutions
        # Specify number of filter after each upconv (last upconv needs to have the same number of channels as input)
        num_filters_upconv = [4] + args.num_filters[:-1]
        for i in xrange(len(args.num_filters)-1,-1,-1):
            bottleneck_output = self._create_bottleneck_layer(upconv_output, args, 'bn_decode'+str(i), args.num_filters[i])
            upconv_output += bottleneck_output
            upconv_output = self._conv_operation(upconv_output, num_filters_upconv[i], 3, args, 'upconv'+str(i), transpose=True, stride=2)

        # Ouput of upconvolutions is reconstructed solution
        self.rec_sol = upconv_output 

    # Create optimizer to minimize loss
    def _create_optimizer(self, args):
        # Find reconstruction loss -- average over batches, sum over points/channels
        reconstruction_errors = tf.losses.mean_squared_error(self.x, self.rec_sol, reduction="none")
        self.loss_reconstruction = tf.reduce_mean(tf.reduce_sum(reconstruction_errors, [1, 2, 3]))

        # Find KL-divergence component of loss
        w_mean, w_logstd = tf.split(self.w_dists, [args.noise_dim, args.noise_dim], axis=1)
        w_std = tf.exp(w_logstd) + 1e-3

        # Define distribution and prior objects
        w_dist = tf.distributions.Normal(loc=w_mean, scale=w_std)
        prior_dist = tf.distributions.Normal(loc=tf.zeros_like(w_mean), scale=tf.ones_like(w_std))
        self.kl_loss = tf.reduce_mean(tf.reduce_sum(tf.distributions.kl_divergence(w_dist, prior_dist), axis=1))

        # Sum with regularization losses to form total cost
        self.cost = self.loss_reconstruction + tf.reduce_sum(tf.losses.get_regularization_losses())+self.kl_weight*self.kl_loss

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))




