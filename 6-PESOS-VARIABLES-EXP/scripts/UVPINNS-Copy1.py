# -*- coding: utf-8 -*-
"""
@version: 1.0.c - Exponential weights
@author : amedrano
@date   : 24/08/2022
@last modified by  : amedrano
@last modified time: 24/08/2022
"""

import tensorflow as tf
import numpy as np
import math
from tqdm import tqdm



class PINN(tf.keras.Model):
    def __init__(self, units = 32, num_layers = 8, num_outputs = 3, alpha=1, beta=1, lambda_exp=1, gamma=1.4):
        super().__init__()
        self.alpha = [alpha]
        self.beta = [beta]
        self.gamma = gamma
        self.lambda_exp = lambda_exp
        self.units = units
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.dense_layers = []
        for i in range(self.num_layers):
            self.dense_layers.append(tf.keras.layers.Dense(self.units, activation = tf.nn.tanh))
        self.dense_output = tf.keras.layers.Dense(self.num_outputs, activation = tf.nn.tanh)
        
        self.layer_sizes = [2]
        for i in range(self.num_layers):
            self.layer_sizes.append(self.units)
        self.layer_sizes.append(self.num_outputs)
        self.layer_sizes.append(self.num_outputs)
        self.sizes_w = []
        self.sizes_b = []
        for i, width in enumerate(self.layer_sizes):
            if i != 1:
                self.sizes_w.append(int(width * self.layer_sizes[1]))
                self.sizes_b.append(int(width if i != 0 else self.layer_sizes[1]))
    
    # Call the object to predict
    def call(self, inputs):
        X = inputs
        for n_layer in range(self.num_layers):
            X = self.dense_layers[n_layer](X)
        X = self.dense_output(X)
        return X
    
    # Calculate the residual for the physics boundaries
    def residual_r(self):
        with tf.GradientTape(persistent=True) as tape:
            t, x = self.X_r[:,0:1], self.X_r[:,1:2]
            tape.watch(t)
            tape.watch(x)
            prediction = self(tf.concat([t, x], axis = 1))
            rho, u, p = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
        rho_x = tape.gradient(rho, x)
        rho_t = tape.gradient(rho, t)
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
        p_x = tape.gradient(p, x)
        p_t = tape.gradient(p, t)
        
        r = (rho_t + u*rho_x + rho*u_x)**2 + (rho *(u_t + u*u_x) + p_x)**2 + (p_t + self.gamma*p*u_x + u*p_x)**2
        return r
    
    # Calculate the residual for the initial conditions
    def residual_ic(self):
        U_pred_initial = self(self.X_0)
        r_ic = (U_pred_initial[:,0:1] - self.U_0[0])**2 + (U_pred_initial[:,1:2] - self.U_0[1])**2 + (U_pred_initial[:,2:3] - self.U_0[2])**2
        return r_ic
    
    # Do one epoch
    def compute_loss(self, X_r, X_0, U_0, optim_w):
        self.X_r = X_r
        self.X_0 = X_0
        self.U_0 = U_0
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.trainable_variables)
            
            # Calcualte the final residual
            loss_r = tf.reduce_mean(self.alpha[-1]*self.residual_r())
            loss_ic = tf.reduce_mean(self.beta[-1]*self.residual_ic())
            loss = loss_r + loss_ic
            
        grad_w = tape.gradient(loss, self.trainable_variables)
        del tape
        
        # Update the weights of the NN:
        optim_w.apply_gradients(zip(grad_w, self.trainable_variables))
        
        return loss, loss_r, loss_ic, grad_w
    
    def train_epochs(self, X_r, X_0, U_0, optim_w, epochs, losses_list, predictions_by_epoch=None):
        loss_hist, loss_r_hist, loss_ic_hist = losses_list
        
        pbar = tqdm(range(epochs))
        for epoch in  pbar:
            # Compute losses
            loss, loss_r, loss_ic, grad_w = self.compute_loss(X_r, X_0, U_0, optim_w)
    
            loss_hist.append(loss.numpy())
            loss_r_hist.append(loss_r.numpy())
            loss_ic_hist.append(loss_ic.numpy())
            
            # Update alpha and beta
            self.alpha.append(self.alpha[-1] * math.exp(self.lambda_exp * loss_r / (loss_r + loss_ic)))
            self.beta.append(self.beta[-1] * math.exp(-self.lambda_exp * loss_ic / (loss_r + loss_ic)))
            
            # Save predicitons for plot
            if isinstance(predictions_by_epoch, list):
                pred = self(tf.concat([t_plot, tf.cast(x_plot, "float32")], axis = 1))
                predictions_by_epoch.append(pred.numpy())
                
            # Update progress bar
            pbar.set_postfix({'Loss': loss.numpy(),
                              'Loss R' : loss_r.numpy(),
                              'Loss IC' : loss_ic.numpy()})