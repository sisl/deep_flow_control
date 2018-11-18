import numpy as np
import tensorflow as tf

class KoopmanModel():
    def __init__(self, args):
        # Placeholder for data -- inputs are number of elements x pts in mesh x dimensionality of data for each point
        self.x = tf.Variable(np.zeros((args.batch_size*(args.seq_length+1), 128, 256, 4), dtype=np.float32), trainable=False, name="input_data")
        if args.control_input:
            self.u = tf.Variable(np.zeros((args.batch_size, args.seq_length, args.action_dim), dtype=np.float32), trainable=False, name="action_values")
        
        # Parameters to be set externally
        self.is_training = tf.Variable(False, trainable=False, name="training_flag")
        self.learning_rate = tf.Variable(0.0, trainable=False, name="learning_rate")

        # Normalization parameters to be stored
        self.shift = tf.Variable(np.zeros(4), trainable=False, name="input_shift")
        self.scale = tf.Variable(np.zeros(4), trainable=False, name="input_scale")
        
        # Create the computational graph
        self._create_encoder(args)
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
    def _create_encoder(self, args):
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
                                    args.code_dim,
                                    name='to_code', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        
        # Reshape output
        self.code_reshape = tf.reshape(code, [args.batch_size, args.seq_length+1, args.code_dim])

        # Get inputs and targets by slicing array
        code_x_reshape = self.code_reshape[:, :args.seq_length]
        code_y_reshape = self.code_reshape[:, 1:]
        self.code_y_reshape = code_y_reshape
        self.code_x = tf.reshape(code_x_reshape, [args.batch_size*args.seq_length, args.code_dim])

        # Determine length of sequence to use in determining A-matrix
        hi_idx = args.seq_length/2 if args.halve_seq else args.seq_length

        # Concatenate code values with control inputs if desired
        if args.control_input:
            # Define B matrix as global variable
            self.B = tf.get_variable("B_matrix", [args.batch_size, args.action_dim, args.code_dim], 
                                                    regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
            code_y_reshape = code_y_reshape - self.u*self.B

        # Find A through least squares
        self.A = tf.matrix_solve_ls(code_x_reshape[:, :hi_idx], code_y_reshape[:, :hi_idx], l2_regularizer=args.l2_regularizer)
        
        # Get predicted code at first time step
        y_pred = tf.matmul(code_x_reshape, self.A)

        # If desired, create recursive predictions for y
        if args.recursive_pred:
            y_pred_t = tf.expand_dims(y_pred[:, 0], axis=1)
            y_pred_list = [y_pred_t]
            for t in xrange(1, args.seq_length):
                y_pred_t = tf.matmul(y_pred_t, self.A)
                
                # Account for control input if necessary
                if args.control_input:
                    u = self.u[:, t]
                    if args.action_dim == 1: u = tf.expand_dims(u, axis=2)
                    y_pred_t += tf.matmul(u, self.B)

                y_pred_list.append(y_pred_t) 
            y_pred = tf.stack(y_pred_list, axis=1)

        # Reshape predicted y
        self.y_pred = tf.reshape(y_pred, [args.batch_size*args.seq_length, args.code_dim])
        self.code_y = tf.reshape(self.code_y_reshape, [args.batch_size*args.seq_length, args.code_dim])

        # Overwrite code with predicted y
        self.code = tf.concat([self.code_x, self.y_pred], axis=0)

    # Decoding stage -- reverse decoder to reconstruct input
    def _create_decoder(self, args):
        # Reverse fully connected layer and reshape
        self.rev_fc_output = tf.layers.dense(self.code, 
                                            self.reshape_output.get_shape().as_list()[-1], 
                                            activation=tf.nn.relu,
                                            name='from_code', 
                                            kernel_regularizer=tf.contrib.layers.l2_regularizer(args.reg_weight))
        conv_shape = self.encoder_conv_output.get_shape().as_list()
        conv_shape[0] = 2*(conv_shape[0]-args.batch_size)
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
        # First find x and y by reshaping, slicing, then reshaping again
        reshape_dim = [args.batch_size, args.seq_length+1, 128, 256, 4]
        input_dim = [args.batch_size*args.seq_length, 128, 256, 4]
            
        # Construct targets
        x_reshape = tf.reshape(self.x, reshape_dim)
        x = tf.reshape(x_reshape[:, :args.seq_length], input_dim)
        y = tf.reshape(x_reshape[:, 1:], input_dim)

        # Find reconstruction loss -- average over batches, sum over points/channels
        reconstruction_errors = tf.losses.mean_squared_error(tf.concat([x, y], axis=0), self.rec_sol, reduction="none")
        reduce_mean_ax = [1, 2, 3]
        self.loss_reconstruction = tf.reduce_mean(tf.reduce_sum(reconstruction_errors, reduce_mean_ax))

        # Sum with regularization losses to form total cost
        self.cost = self.loss_reconstruction + tf.reduce_sum(tf.losses.get_regularization_losses())

        # Perform parameter update
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables()
        self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        self.train = optimizer.apply_gradients(zip(self.grads, tvars))

        

    