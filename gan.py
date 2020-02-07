import os

os.environ['PYTHONHASHSEED'] = '0'

from tensorflow import set_random_seed as tf_set_random_seed

tf_set_random_seed(1234)

import numpy as np

np.random.seed(42)

import random

random.seed(7)

# Can turn off GPU on CPU-only machines; maybe results in faster startup.
use_GPU = False
if use_GPU is False:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from constants import *
from gan_plots import *

import tensorflow as tf
import numpy as np
import pickle
import math
from imageio import imread, mimsave
from functools import partial
from random import randint

# use this because switched from AAA bonds to NASDAQ
VAR2 = "NASDAQ"


class VIXMatHandler:
    def __init__(self, vix_mat):
        self.vix_mat_shape = vix_mat.shape

        self.vix_mat_mean = np.mean(vix_mat)
        self.vix_mat_std = np.std(vix_mat)

        self.vix_mat_stdz = (vix_mat - self.vix_mat_mean) / self.vix_mat_std

        self.vix_mat_total = self.vix_mat_stdz

    def simulated_vix(self, center=None, dispersion=None, shape=None):
        if center is None:
            center = self.vix_mat_mean
        if dispersion is None:
            dispersion = self.vix_mat_std
        if shape is None:
            shape = self.vix_mat_shape

        rnd = np.random.normal(loc=center, scale=dispersion, size=shape)

        max_val = center + dispersion
        rnd[rnd > max_val] = max_val

        min_val = center - dispersion
        rnd[rnd < min_val] = min_val

        rnd_stdz = (rnd - self.vix_mat_mean) / self.vix_mat_std

        return rnd_stdz


class Batches:
    def __init__(self, data):
        self.n = data.shape[0]
        self.n_batches = math.ceil(self.n / BATCH_SIZE)
        self.shuffle()

    def shuffle(self):
        self.idx = np.arange(0, self.n)
        np.random.shuffle(self.idx)

        self.index_list = list()

        if self.n <= BATCH_SIZE:
            self.index_list.append(self.idx)
        else:
            for i in range(self.n_batches):
                start = i * BATCH_SIZE
                stop = min(self.n, (i + 1) * BATCH_SIZE)
                self.index_list.append(self.idx[start:stop])


def generate_random(shape):
    rand = np.random.normal(loc=RAND_MEAN, scale=RAND_STD, size=shape)
    return rand


def sample_matrix(x, vix, draw_correl_charts=False):
    x_len = x.shape[0]
    num_rows = x_len - SAMPLE_LEN + 1

    x_mat = np.zeros((num_rows, N_SERIES, SAMPLE_LEN))
    vix_mat = np.zeros((num_rows, N_CONDITIONALS, SAMPLE_LEN))

    correls = np.zeros((num_rows, N_SERIES))

    for n in range(num_rows):
        col0 = x[n:(n + SAMPLE_LEN), 0]
        col1 = x[n:(n + SAMPLE_LEN), 1]
        col2 = x[n:(n + SAMPLE_LEN), 2]

        vix_sample = vix[n:(n + SAMPLE_LEN)]

        correls[n, 0] = np.corrcoef(col0, col1)[0, 1]
        correls[n, 1] = np.corrcoef(col0, col2)[0, 1]
        correls[n, 2] = np.corrcoef(col1, col2)[0, 1]

        x_mat[n, 0, :] = col0
        x_mat[n, 1, :] = col1
        x_mat[n, 2, :] = col2

        vix_mat[n, 0, :] = vix_sample

    if draw_correl_charts:
        dir = ['../images/base_correlations/']
        delete_files_in_folder(dir[0])
        hist_elements = [{'data': correls[:, 0], f'label': 'R3000 vs. {VAR2}'}, {'data': correls[:, 1], 'label': 'R3000 vs. EM HY'},
                         {'data': correls[:, 2], f'label': '{VAR2} vs. EM HY'}]
        f_name = f'correls_actual.png'
        dist_chart(hist_elements, 'Correlations', 'Frequency', 'Actual data correlations (1999 to 2018)', f_name=f_name, bins=30,
                   directories=dir, scaleX=[-1., 1.])

    return x_mat, correls, vix_mat


def load_data():
    if USE_SAVED_X:
        x = pickle.load(open(f'{X_SAVE_PATH}last_x.p', 'rb'))
        x_mat = pickle.load(open(f'{X_SAVE_PATH}last_x_mat.p', 'rb'))
        correls_actual = pickle.load(open(f'{X_SAVE_PATH}last_correls_actual.p', 'rb'))
        vix = pickle.load(open(f'{X_SAVE_PATH}last_vix.p', 'rb'))
        vix_mat = pickle.load(open(f'{X_SAVE_PATH}last_vix_mat.p', 'rb'))
    else:
        from weekly_returns import weekly_returns as x
        from weekly_returns import vix
        x_mat, correls_actual, vix_mat = sample_matrix(x, vix, draw_correl_charts=True)

        pickle.dump(x, open(f'{X_SAVE_PATH}last_x.p', 'wb'))
        pickle.dump(x_mat, open(f'{X_SAVE_PATH}last_x_mat.p', 'wb'))
        pickle.dump(correls_actual, open(f'{X_SAVE_PATH}last_correls_actual.p', 'wb'))
        pickle.dump(vix, open(f'{X_SAVE_PATH}last_vix.p', 'wb'))
        pickle.dump(vix_mat, open(f'{X_SAVE_PATH}last_vix_mat.p', 'wb'))

    return x, x_mat, correls_actual, vix, vix_mat


def get_batch_inputs(X_batch, x_mat_normalized, vix_handler, rnd):
    n_batches = X_batch.n_batches
    b = randint(0, n_batches - 1)

    idx = X_batch.index_list[b]

    X_in = x_mat_normalized[idx]
    Z_in = rnd[idx]
    C_in = vix_handler.vix_mat_total[idx]

    return X_in, Z_in, C_in


def main(start_model=None, saved_model_dir=None, gen_encourage=1.0, aggregate_models=False, shuffle=False):
    x, x_mat, correls_actual, vix, vix_mat = load_data()

    vix_handler = VIXMatHandler(vix_mat)

    x_mat_mean = np.mean(x_mat)
    x_mat_std = np.std(x_mat)
    x_mat_normalized = (x_mat - x_mat_mean) / x_mat_std

    x_mat_split = np.split(x_mat, indices_or_sections=N_SERIES, axis=1)
    x_mat_means = list()
    x_mat_stdevs = list()

    for m in x_mat_split:
        mm = np.squeeze(m, 1)
        x_mat_means.append(np.mean(mm, axis=1) * 100.)
        x_mat_stdevs.append(np.std(mm, axis=1) * 100.)

    tf.reset_default_graph()

    from rnn_static import create_rnn_discriminator, create_rnn_generator

    discriminator = partial(create_rnn_discriminator)
    generator = partial(create_rnn_generator)

    X = tf.placeholder(tf.float32, [None, N_SERIES, SAMPLE_LEN])
    Z = tf.placeholder(tf.float32, [None, N_SERIES, SAMPLE_LEN])
    C = tf.placeholder(tf.float32, [None, N_CONDITIONALS, SAMPLE_LEN])

    g_out, g_train = generator(Z=Z, conditionals=C)
    x_out, x_train = discriminator(X=X, conditionals=C)
    z_out, z_train = discriminator(X=g_out, conditionals=C)

    disc_loss_base = -tf.reduce_mean(x_out) + tf.reduce_mean(z_out)
    gen_loss_base = -tf.reduce_mean(z_out)

    # discriminator gradient penalty
    scale = 10.
    epsilon = tf.random_uniform([], minval=0., maxval=1.)
    x_h = epsilon * X + (1. - epsilon) * g_out

    disc_for_grad, training_flag_for_grad = discriminator(X=x_h, conditionals=C)
    grad_d_x_h = tf.gradients(disc_for_grad, x_h)[0]

    grad_d_x_h = tf.contrib.layers.flatten(grad_d_x_h)
    print('grad_d_x_h')
    print(grad_d_x_h)

    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_d_x_h), axis=1))
    grad_pen = tf.reduce_mean(tf.square(grad_norm - 1.))

    disc_loss_base += scale * grad_pen

    # generator gradient encouragement
    generator_variation_scaler = gen_encourage
    generator_output, gen_training_flag_for_grad = generator(Z=Z, conditionals=C)
    grad_g_z = tf.gradients(generator_output, Z)[0]

    grad_g_z = tf.contrib.layers.flatten(grad_g_z)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_g_z), axis=1))

    # try to smooth gradient benefit a bit
    grad_norm_sqrt = tf.sqrt(tf.sqrt(grad_norm))
    grad_norm_sqrt = tf.reduce_mean(grad_norm_sqrt)

    gen_loss_base -= generator_variation_scaler * grad_norm_sqrt

    if L2_REGULARIZATION > 0.:
        gen_l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=GENERATOR_SCOPE)
        disc_l2_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=DISCRIMINATOR_SCOPE)

        gen_loss = tf.add_n([gen_loss_base] + gen_l2_loss)
        disc_loss = tf.add_n([disc_loss_base] + disc_l2_loss)
    else:
        gen_loss = gen_loss_base
        disc_loss = disc_loss_base

    gen_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.0, decay=0.9, epsilon=1e-10)
    disc_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4, momentum=0.0, decay=0.9, epsilon=1e-10)

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=GENERATOR_SCOPE)
    gen_step = gen_opt.minimize(gen_loss, var_list=gen_vars)

    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=DISCRIMINATOR_SCOPE)
    disc_step = disc_opt.minimize(disc_loss, var_list=disc_vars)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    save_dir_all = "../model_saves/all/"

    png_files_01 = list()
    png_files_02 = list()
    png_files_12 = list()

    png_moments_files = [list(), list(), list()]

    png_points_files = [list(), list(), list()]

    generated_values = list()

    losses_list = list()

    with tf.Session() as sess:

        if saved_model_dir is not None:

            if shuffle:
                dir_append = '_shuffled'
            else:
                dir_append = ''

            if aggregate_models:
                model_list = list(range(MIN_EPOCHS + GRAPH_STEP, EPOCHS + GRAPH_STEP, GRAPH_STEP))
                dir_name = f'{RESULTS_USED_DIR}simulated_results/agg_models/2008{dir_append}/'
            else:
                model_list = [EPOCHS]
                dir_name = f'{RESULTS_USED_DIR}simulated_results/last_model/2008{dir_append}/'  # dir_name = f'{
                # RESULTS_USED_DIR}simulated_results/last52weeks_shuffled/'

            n_models = len(model_list)

            num_batches = 1
            num_sims = num_batches * BATCH_SIZE

            results = np.zeros((n_models, num_sims, N_SERIES, SAMPLE_LEN))

            kp_prob = KEEP_PROB if GENERATOR_DROPOUT_ALWAYS_ON else 1.

            from weekly_returns import weekly_returns as x_test_import
            from weekly_returns import vix as vix_test_import

            x_test = np.copy(x_test_import)
            vix_test = np.copy(vix_test_import)

            x_test = np.swapaxes(x_test, 0, 1)
            x_test = np.expand_dims(x_test, axis=0)

            vix_test = np.expand_dims(vix_test, axis=1)
            vix_test = np.expand_dims(vix_test, axis=0)
            vix_test = np.swapaxes(vix_test, 1, 2)
            vix_test = np.tile(vix_test, (BATCH_SIZE, 1, 1))

            # last 52 weeks: [:, :, -52:]
            # 2008: [:, :, 460:512]
            #
            x_test = x_test[:, :, 460:512]
            vix_test = vix_test[:, :, 460:512]

            vix_test = (vix_test - vix_handler.vix_mat_mean) / vix_handler.vix_mat_std

            for m, model in enumerate(model_list):
                model_dir = f'{saved_model_dir}{model}/model.ckpt'
                saver.restore(sess, model_dir)

                for sim in range(num_batches):
                    rnd = generate_random((BATCH_SIZE, N_SERIES, SAMPLE_LEN))

                    genval = sess.run(g_out, feed_dict={
                        Z: rnd, C: vix_test, g_train: kp_prob, x_train: kp_prob, z_train: kp_prob
                        })

                    genval_unnormalized = (genval * x_mat_std) + x_mat_mean

                    print('x:')
                    print(x_test)
                    print('genval:')
                    print(genval)
                    print('genval unnormalized:')
                    print(genval_unnormalized)

                    results[m, sim * BATCH_SIZE:(sim + 1) * BATCH_SIZE] = genval_unnormalized

                if shuffle:
                    for time in range(SAMPLE_LEN):
                        idx_list = list(range(num_sims))
                        random.shuffle(idx_list)
                        results[m, :, :, time] = results[m, idx_list, :, time]

            results = np.reshape(results, (n_models * num_sims, N_SERIES, SAMPLE_LEN))

            line_width = 0.3
            labels = ['R3000', VAR2, 'EMHY']
            colors = [('#00BFFFff', '#c32148ff'), ('#fd5e0fff', '#228b22ff'), ('#daa520ff', '#b710aaff')]

            for s, series in enumerate(labels):
                f_name = f'simulated_{series}_genScale_{str(round(gen_scale * 100))}.png'

                data_sets = list()

                generated_to_plot = results[:, s, :]
                data_sets.append(generated_to_plot)

                observed_to_plot = x_test[:, s, :]
                data_sets.append(observed_to_plot)

                cumul_return_plot(data_sets=data_sets, f_name=f_name, dir_name=dir_name, legend_loc=3, labels=('actual', 'generated'),
                                  colors=(colors[s][1], colors[s][0]), title=f'{series} Cumulative Returns: Actual vs. Simulated',
                                  sizes=(line_width, 5.))

                np.savetxt(f'{dir_name}_{series}_generated_genScale_{str(round(gen_scale * 100))}.txt', generated_to_plot, delimiter=', ')
                np.savetxt(f'{dir_name}_{series}_actual_genScale_{str(round(gen_scale * 100))}.txt', observed_to_plot, delimiter=', ')

            pairs = ([0, 1], [0, 2], [1, 2])
            ylim = ([0.82, 1.], [-0.25, 1.], [-0.25, 1.])
            float_format = ('%.2f', '%.1f', '%.1f')

            correl_len = 24

            cond_gen = np.zeros((num_sims, SAMPLE_LEN - correl_len + 1))
            cond_obs = np.zeros((1, SAMPLE_LEN - correl_len + 1))

            start = 0
            stop = start + correl_len

            while stop <= SAMPLE_LEN:

                for n in range(num_sims):

                    gen_mat = results[n, :, start:stop].T
                    obs_mat = x_test[0, :, start:stop].T

                    gen_mat_flat = gen_mat.flatten('F')
                    gen_mat_len = np.sqrt(np.sum(np.power(gen_mat_flat, 2.)))
                    gen_mat_scaled = gen_mat / gen_mat_len

                    obs_mat_flat = obs_mat.flatten('F')
                    obs_mat_len = np.sqrt(np.sum(np.power(obs_mat_flat, 2.)))
                    obs_mat_scaled = obs_mat / obs_mat_len

                    XtX_gen = np.matmul(gen_mat_scaled.T, gen_mat_scaled)
                    XtX_obs = np.matmul(obs_mat_scaled.T, obs_mat_scaled)

                    Eigs_gen = np.abs(np.linalg.eigvals(XtX_gen))
                    cond_gen[n, start] = np.sqrt(np.max(Eigs_gen) / np.min(Eigs_gen))

                    Eigs_obs = np.abs(np.linalg.eigvals(XtX_obs))
                    cond_obs[0, start] = np.sqrt(np.max(Eigs_obs) / np.min(Eigs_obs))

                stop = stop + 1
                start = start + 1

            data_sets = list()

            data_sets.append(cond_gen)
            data_sets.append(cond_obs)

            line_plot(data_sets=data_sets, f_name=f'condition_numbers_genScale_{str(round(gen_scale * 100))}.png', dir_name=dir_name,
                      labels=['Generated', 'Actual'], legend_loc=2, colors=(colors[0][1], colors[0][0]), sizes=(line_width * 2., 5.),
                      title=f'Actual vs. Generated Multicorrelation', ylabel='Condition Number', xlabel='Time')

            np.savetxt(f'{dir_name}_condition_index_generated_{str(round(gen_scale * 100))}.txt', cond_gen, delimiter=', ')
            np.savetxt(f'{dir_name}_condition_index_actual_{str(round(gen_scale * 100))}.txt', cond_obs, delimiter=', ')

            for p, pair in enumerate(pairs):
                f_name_base = f'correlations_{labels[pair[0]]}_{labels[pair[1]]}_genScale_{str(round(gen_scale * 100))}'

                data_sets = list()

                corr_gen = np.zeros((num_sims, SAMPLE_LEN - correl_len + 1))
                corr_observed = np.zeros((1, SAMPLE_LEN - correl_len + 1))
                start = 0
                stop = start + correl_len
                while stop <= SAMPLE_LEN:

                    for n in range(num_sims):
                        gen_mat_0 = results[n, pair[0], start:stop]
                        gen_mat_1 = results[n, pair[1], start:stop]

                        gen_mat_0 = np.expand_dims(gen_mat_0, axis=0)
                        gen_mat_1 = np.expand_dims(gen_mat_1, axis=0)

                        gen_correl_mat = np.corrcoef(np.vstack((gen_mat_0, gen_mat_1)))
                        corr_gen[n, start] = gen_correl_mat[1, 0]

                        obs_mat_0 = x_test[0, pair[0], start:stop]
                        obs_mat_1 = x_test[0, pair[1], start:stop]

                        obs_mat_0 = np.expand_dims(obs_mat_0, axis=0)
                        obs_mat_1 = np.expand_dims(obs_mat_1, axis=0)

                        obs_correl_mat = np.corrcoef(np.vstack((obs_mat_0, obs_mat_1)))
                        corr_observed[0, start] = obs_correl_mat[1, 0]

                    stop = stop + 1
                    start = start + 1

                data_sets.append(corr_gen)
                data_sets.append(corr_observed)

                np.savetxt(f'{dir_name}_generated_{f_name_base}.txt', corr_gen, delimiter=', ')
                np.savetxt(f'{dir_name}_actual_{f_name_base}.txt', corr_observed, delimiter=', ')

                line_plot(data_sets=data_sets, f_name=f'{f_name_base}.png', dir_name=dir_name, labels=['Generated', 'Actual'], legend_loc=3,
                          colors=(colors[p][1], colors[p][0]), sizes=(line_width * 2., 5.),
                          title=f'{labels[pair[0]]} and {labels[pair[1]]}: Actual vs. Generated Rolling Correlations', ylabel='Correlation',
                          xlabel='Time', ylim=ylim[p], float_format=float_format[p])

            return None

        init.run()

        start_epoch = 1
        end_epoch = EPOCHS

        if start_model is not None:
            start_epoch = start_model['start_epoch']
            saver.restore(sess, start_model['start_model_dir'])

            graph_steps = list(range(GRAPH_STEP, start_epoch, GRAPH_STEP))
            graph_steps = [1] + graph_steps

            for g in graph_steps:
                f_name_01 = f'R3000vAAA_epoch{g}.png'
                f_name_02 = f'R3000vEMHY_epoch{g}.png'
                f_name_12 = f'AAAvEMHY_epoch{g}.png'

                png_files_01.append(f_name_01)
                png_files_02.append(f_name_02)
                png_files_12.append(f_name_12)

                f_names = ['R3000', VAR2, 'EMHY']

                for s in range(N_SERIES):
                    file_name = f'{f_names[s]}_epoch{g}.png'
                    png_moments_files[s].append(file_name)
                    png_points_files[s].append(file_name)

        for epch in range(start_epoch, end_epoch + 1):

            if epch > 1:
                # The generator gets a new set of random data every epoch.
                # This is because one motivation for a GAN is to generate an unlimited number of samples.
                rnd = generate_random(x_mat_normalized.shape)

                X_batch = Batches(x_mat_normalized)
                n_batches = X_batch.n_batches

                g_losses = list()
                d_losses = list()

                for b in range(n_batches):
                    for t in range(DISCRIMINATOR_TRAINS_PER_BATCH):
                        X_in, Z_in, C_in = get_batch_inputs(X_batch, x_mat_normalized, vix_handler, rnd)

                        _, dloss = sess.run([disc_step, disc_loss], feed_dict={
                            X: X_in, Z: Z_in, C: C_in, g_train: KEEP_PROB, x_train: KEEP_PROB, z_train: KEEP_PROB
                            })

                        d_losses.append(dloss)

                    for t in range(GENERATOR_TRAINS_PER_BATCH):
                        X_in, Z_in, C_in = get_batch_inputs(X_batch, x_mat_normalized, vix_handler, rnd)

                        _, gloss = sess.run([gen_step, gen_loss], feed_dict={
                            Z: Z_in, C: C_in, g_train: KEEP_PROB, x_train: KEEP_PROB, z_train: KEEP_PROB
                            })

                        gge = sess.run([gen_grad_encourage], feed_dict={
                            Z: Z_in, C: C_in, g_train: KEEP_PROB, x_train: KEEP_PROB, z_train: KEEP_PROB
                            })
                        print(f"gen grad encourage: {gge}")

                        g_losses.append(gloss)

                print(d_losses)
                print(g_losses)

                gen_mean_loss = np.array(g_losses).mean()
                disc_mean_loss = np.array(d_losses).mean()

                loss_string = f"Iter: {epch}   Disc_loss: {disc_mean_loss}   Gen_loss: {gen_mean_loss}"
                print(loss_string)
                losses_list.append(loss_string)

            save_charts = False

            if epch == 1 or epch % GRAPH_STEP == 0 or epch == EPOCHS:
                save_charts = True

            if save_charts:

                if epch >= MIN_EPOCHS:
                    saver.save(sess, f"{save_dir_all}epoch_{epch}/model.ckpt")

                rnd_validation = np.random.normal(loc=RAND_MEAN, scale=RAND_STD, size=x_mat_normalized.shape)
                Z_val_batch = Batches(rnd_validation)

                genval = np.zeros(x_mat_normalized.shape)

                kp_prob = KEEP_PROB if GENERATOR_DROPOUT_ALWAYS_ON else 1.

                for b in range(Z_val_batch.n_batches):
                    idx = Z_val_batch.index_list[b]

                    genval[b * BATCH_SIZE:(b + 1) * BATCH_SIZE] = sess.run(g_out, feed_dict={
                        Z: rnd_validation[idx], C: vix_handler.vix_mat_total[idx], g_train: kp_prob, x_train: kp_prob, z_train: kp_prob
                        })

                # plot distributions
                genval_unnormalized = (genval * x_mat_std) + x_mat_mean
                genval_unnormalized_split_temp = np.split(genval_unnormalized, indices_or_sections=N_SERIES, axis=1)

                genval_unnormalized_split = list()
                for gust in genval_unnormalized_split_temp:
                    genval_unnormalized_split.append(np.squeeze(gust, axis=1))

                f_names = ['R3000', VAR2, 'EMHY']
                title_names = ['Russell 3000', VAR2, 'Emerging markets high yield']

                size = 80

                # plot mean/stdev
                gen_unnrml_series_means = list()
                gen_unnrml_series_stdevs = list()

                for g in genval_unnormalized_split:
                    gen_unnrml_series_means.append(np.mean(g, axis=1) * 100.)
                    gen_unnrml_series_stdevs.append(np.std(g, axis=1) * 100.)

                directories = ['../images/pngs_moments/', '../images/pngs_moments_for_gif/']
                # colors = ('#7897e6ff', '#ab4448ff', '#488f59ff', '#daa520ff', '#FD5E0F', '#966fd6ff')
                # colors = ('#55a868ff', '#c44e52ff', '#85a8ffff', '#FFD700ff', '#FD5E0Fff', '#7897e6ff')
                # colors = ('#779ecbff', '#C23B23ff', '#03C03Cff', '#f56f02ff', '#966fd6ff', '#c3c343ff')
                colors = ('#00BFFFff', '#c32148ff', '#fd5e0fff', '#228b22ff', '#daa520ff', '#b710aaff')
                for s in range(N_SERIES):
                    elements = [{
                        'x': x_mat_means[s], 'y': x_mat_stdevs[s], 'label': 'actual', 'alpha': 1., 'color': colors[s * 2], 'size': size
                        }, {
                        'x': gen_unnrml_series_means[s], 'y': gen_unnrml_series_stdevs[s], 'label': 'generated', 'alpha': 1.,
                        'color': colors[s * 2 + 1], 'size': size
                        }]

                    file_name = f'{f_names[s]}_epoch{epch}.png'
                    scatter(elements, 'Weekly mean return (%)', 'Weekly standard deviation (%)',
                            f'{title_names[s]}: actual vs. generated dispersion (epoch {epch})', save_file=file_name,
                            directories=directories)

                    png_moments_files[s].append(file_name)

                # plot points
                directories = ['../images/pngs_distributions/', '../images/pngs_distributions_for_gif/']
                for s in range(N_SERIES):
                    elements = [{'data': x_mat_split[s].flatten() * 100., 'label': f'actual'}, {
                        'data': genval_unnormalized_split[s].flatten() * 100., 'label': f'generated'
                        }]

                    file_name = f'{f_names[s]}_epoch{epch}.png'
                    title = f'{title_names[s]}: actual vs. generated data (epoch {epch})'
                    dist_chart(elements, 'Weekly returns (%)', 'Frequency', title=title, f_name=file_name, bins=100, color_start=s * 2,
                               directories=directories)

                    png_points_files[s].append(file_name)

                # plot correlations
                gen_series_temp = np.split(genval, indices_or_sections=N_SERIES, axis=1)

                gen_series = list()
                for gst in gen_series_temp:
                    gen_series.append(np.squeeze(gst, axis=1))

                nr = gen_series[0].shape[0]

                corr_01 = np.zeros((nr, 1))
                corr_02 = np.zeros((nr, 1))
                corr_12 = np.zeros((nr, 1))

                for r in range(nr):
                    corr_01[r, 0] = np.corrcoef(gen_series[0][r], gen_series[1][r])[0, 1]
                    corr_02[r, 0] = np.corrcoef(gen_series[0][r], gen_series[2][r])[0, 1]
                    corr_12[r, 0] = np.corrcoef(gen_series[1][r], gen_series[2][r])[0, 1]

                generated_values.append({
                    'epoch': epch, 'genval': genval, 'corr_01': corr_01, 'corr_02': corr_02, 'corr_12': corr_12
                    })

                x_lab = 'Correlations'
                y_lab = 'Frequency'
                bins = 30
                directories = ['../images/pngs_correls/', '../images/pngs_correls_for_gif/']
                scaleX = [-1., 1.]

                title = f'Russell 3000 vs. {VAR2}: actual and generated correlations (epoch {epch})'
                f_name_01 = f'R3000vAAA_epoch{epch}.png'
                el = [{'data': correls_actual[:, 0], 'label': 'actual'}, {'data': corr_01, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_01, bins=bins, color_start=0, directories=directories, scaleX=[0.5, 1.])

                title = f'Russell 3000 vs. Emerging markets HY: actual and generated correlations (epoch {epch})'
                f_name_02 = f'R3000vEMHY_epoch{epch}.png'
                el = [{'data': correls_actual[:, 1], 'label': 'actual'}, {'data': corr_02, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_02, bins=bins, color_start=2, directories=directories, scaleX=scaleX)

                title = f'{VAR2} vs. Emerging markets HY: actual and generated correlations (epoch {epch})'
                f_name_12 = f'AAAvEMHY_epoch{epch}.png'
                el = [{'data': correls_actual[:, 2], 'label': 'actual'}, {'data': corr_12, 'label': 'generated'}]
                dist_chart(el, x_lab, y_lab, title, f_name_12, bins=bins, color_start=4, directories=directories, scaleX=scaleX)

                png_files_01.append(f_name_01)
                png_files_02.append(f_name_02)
                png_files_12.append(f_name_12)

        with open(LOSSES_PATH, 'w') as loss_file:
            for ls in losses_list:
                loss_file.write(f"{ls}\n")

        gif_names = ['01', '02', '12']
        gif_vars = [png_files_01, png_files_02, png_files_12]

        for i, gv in enumerate(gif_vars):
            images_dist_plot = list()
            for f in gv:
                img = imread('../images/pngs_correls_for_gif/' + f)
                images_dist_plot.append(img)
            mimsave(f'../images/gifs/correlations_{gif_names[i]}.gif', images_dist_plot, duration=1.)

        gif_names = ['R3000', VAR2, 'EMHY']
        for i, dist in enumerate(png_moments_files):
            images_dist_plot = list()
            for f in dist:
                img = imread('../images/pngs_moments_for_gif/' + f)
                images_dist_plot.append(img)
            mimsave(f'../images/gifs/moments_{gif_names[i]}.gif', images_dist_plot, duration=1.)

        for i, dist in enumerate(png_points_files):
            images_dist_plot = list()
            for f in dist:
                img = imread('../images/pngs_distributions_for_gif/' + f)
                images_dist_plot.append(img)
            mimsave(f'../images/gifs/distributions_{gif_names[i]}.gif', images_dist_plot, duration=1.)


if __name__ == '__main__':

    # # MODEL TRAINING
    # #
    # saved_model_dir = None
    # # saved_model_dir = f'{RESULTS_USED_DIR}model_saves/all/epoch_{EPOCHS}/model.ckpt'
    #
    # start_model = None
    # # start_model = {'start_epoch': 1001, 'start_model_dir': '../model_saves/all/epoch_1000/model.ckpt'}
    #
    # if saved_model_dir is None and start_model is None:
    #     delete_files_in_folder('../model_saves/all/')
    #     delete_files_in_folder('../images/pngs_correls/')
    #     delete_files_in_folder('../images/pngs_correls_for_gif/')
    #     delete_files_in_folder('../images/gifs/')
    #     delete_files_in_folder('../images/pngs_moments/')
    #     delete_files_in_folder('../images/pngs_moments_for_gif/')
    #     delete_files_in_folder('../images/pngs_distributions/')
    #     delete_files_in_folder('../images/pngs_distributions_for_gif/')
    #
    # if os.path.isfile(LOSSES_PATH):
    #     os.remove(LOSSES_PATH)
    #
    # main(start_model=start_model, saved_model_dir=saved_model_dir, gen_encourage=7.0)

    # # CREATE GIF
    # #
    # create_moments_gif()

    # SIMULATIONS
    #
    gen_scale = (
        (0., '0'), (0.25, '0pt25'), (0.5, '0pt50'), (0.75, '0pt75'), (1., '1'), (1.5, '1pt50'), (2., '2'), (3., '3'), (4., '4'), (5.,
        '5'),
        (6., '6'), (7., '7'))

    for shuffle in [True, False]:
        for gs in gen_scale:
            RESULTS_USED_DIR = f'../results_saves/NASDAQ/2sqrt_gen_encourage_{gs[1]}_2layers_64g_64d/'
            saved_model_dir = f'{RESULTS_USED_DIR}model_saves/all/epoch_'

            main(start_model=None, saved_model_dir=saved_model_dir, gen_encourage=gs[0], aggregate_models=True, shuffle=shuffle)
