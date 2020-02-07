from constants import *

import tensorflow as tf
import numpy as np
from functools import partial

XAV_INIT = tf.contrib.layers.xavier_initializer()


def rnn_graph(X, conditionals, scope, rnn_units):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        with tf.variable_scope(f"{scope}_input"):

            keep_prob = tf.placeholder_with_default(KEEP_PROB, shape=())

            if UNCONDITIONAL:
                X_all = X
            else:
                X_all = tf.concat([X, conditionals], axis=1)

            X_all2 = tf.transpose(X_all, perm=[2, 0, 1])
            X_sequences = tf.unstack(X_all2)

        with tf.variable_scope(f"{scope}_rnn"):

            RNN_cell = partial(tf.contrib.rnn.GRUCell, activation=tf.nn.tanh, kernel_initializer=XAV_INIT)
            cells = [RNN_cell(num_units=neurons) for neurons in rnn_units]

            if DROPOUT > 0.0:
                cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) for cell in cells]
                multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
            else:
                multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)

            initial_states_initializer = tf.random_uniform_initializer(minval=-0.1, maxval=0.1)
            initial_states = list()

            for u, units in enumerate(rnn_units):
                init_state = tf.get_variable(f'{scope}_init_state_{u}', shape=[1, units], initializer=initial_states_initializer,
                                             dtype=tf.float32, trainable=True)

                init_state_batched = tf.tile(init_state, [BATCH_SIZE, 1])

                initial_states.append(init_state_batched)

            rnn_outputs, states = tf.contrib.rnn.static_rnn(multi_layer_cell, X_sequences, dtype=tf.float32,
                                                            initial_state=tuple(initial_states))

        if scope == GENERATOR_SCOPE:
            with tf.variable_scope("generator_output"):

                dense_out_final_list = list()

                for r, output in enumerate(rnn_outputs):
                    with tf.variable_scope(f"generator_output_dense"):
                        dense_out_final = tf.layers.dense(output, units=N_SERIES, kernel_initializer=XAV_INIT, kernel_regularizer=None,
                                                          activation=None)
                    dense_out_final_list.append(dense_out_final)

                out = tf.transpose(dense_out_final_list, perm=[1, 2, 0])

        else:
            with tf.variable_scope('discriminator_output'):

                # compatibility with previous graph version
                end_state = states[-1]
                end_state_final = end_state

                print(' ')
                if DROPOUT > 0.0:
                    drop = tf.layers.dropout(inputs=end_state_final, rate=1.0 - keep_prob, training=True)
                    out = tf.layers.dense(drop, units=1, kernel_initializer=XAV_INIT, kernel_regularizer=None, activation=None)
                else:
                    out = tf.layers.dense(end_state_final, units=1, kernel_initializer=XAV_INIT, kernel_regularizer=None, activation=None)

                for p in [end_state_final]:
                    print(p)

        for p in [X_all, X_all2, X_sequences, rnn_outputs, states, out]:
            print(p)

        # Count trainable parameters in graph.
        n_trainable_params = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope=scope)])
        print(f'{scope} trainable parameters: {n_trainable_params}')

        # Add optional L2 regularization.
        if L2_REGULARIZATION > 0.:
            l2 = sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables(scope=scope) if not ("Bias" in tf_var.name))
            regularization_cost = L2_REGULARIZATION * l2
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, value=regularization_cost)

    return out, keep_prob


def create_rnn_generator(Z, conditionals):
    out, keep_prob = rnn_graph(X=Z, conditionals=conditionals, scope=GENERATOR_SCOPE, rnn_units=[64, 64])
    return out, keep_prob


def create_rnn_discriminator(X, conditionals):
    out, keep_prob = rnn_graph(X=X, conditionals=conditionals, scope=DISCRIMINATOR_SCOPE, rnn_units=[64, 64])
    return out, keep_prob
