# -*- coding: utf-8 -*-
"""
@version: 1.0
@author : amedrano
@date   : 24/08/2022
@last modified by  : amedrano
@last modified time: 24/08/2022
"""

import tensorflow as tf
import numpy as np

# Initial conditions of the problem
def fun_U_0(x, DTYPE = "float32"):
    x_numpy = x.numpy()
    ic_rho = lambda x: 1.0 * (x <= 0.5) + 0.125 * (x > 0.5)
    ic_u = lambda x: 0. * (x <= 0.5) + 0. *(x > 0.5)
    ic_p = lambda x: 1.0 * (x <= 0.5) + 0.1 * (x > 0.5)
    
    return [tf.constant(ic_rho(x_numpy), dtype = DTYPE),
            tf.constant(ic_u(x_numpy), dtype = DTYPE),
            tf.constant(ic_p(x_numpy), dtype = DTYPE)]


def create_dataset_r(xmin, xmax, tmin, tmax, N_r, seed):
    # Set random seed for reproducible results.
    tf.random.set_seed(seed)

    # Draw uniform sample points for initial boundary data.
    x_space = tf.random.uniform(minval = xmin, maxval = xmax, shape = (N_r,1))
    t_space = tf.random.uniform(minval = tmin, maxval = tmax, shape = (N_r,1))
    
    X_r = tf.concat([t_space, x_space], axis = 1)
    
    return X_r


def create_dataset_ic(xmin, xmax, tmin ,N_0, DTYPE="float32"):
    t_0 = tf.reshape(tf.constant(np.repeat(tmin, N_0), dtype = DTYPE), shape = (-1,1))
    x_0 = tf.random.uniform(minval = xmin, maxval = xmax, shape = (N_0,1))
    X_0 = tf.concat([t_0, x_0], axis = 1)
    U_0 = fun_U_0(X_0[:,1:2])

    return X_0, U_0


def create_dataset(xmin, xmax, tmin, tmax, N_r, N_0, seed, DTYPE="float32"):
    X_r = create_dataset_r(xmin, xmax, tmin, tmax, N_r, seed)
    X_0, U_0 = create_dataset_ic(xmin, xmax, tmin, N_0, DTYPE=DTYPE)
    
    return X_r, X_0, U_0