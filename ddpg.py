#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2019/8/11

"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np

LR_A = 0.001  # learning rate for actor
LR_C = 0.002  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 256


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, model, retrain):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.model = model
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.retrain = retrain

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=4)
        ckpt = tf.train.get_checkpoint_state('./' + model)
        if ckpt and ckpt.model_checkpoint_path and not self.retrain:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def save(self, episode):
        self.saver.save(self.sess, self.model + '/' + self.model, global_step=episode)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            hidden = tf.layers.dense(s, 30, activation=tf.nn.relu, kernel_initializer=init_w,
                                     bias_initializer=init_b, name='l1', trainable=trainable)
            # a = tf.layers.dense(net, self.a_dim, name='a', trainable=trainable)
            hidden = tf.layers.dense(hidden, 60, activation=tf.nn.relu, kernel_initializer=init_w,
                                     bias_initializer=init_b, name='a1', trainable=trainable)
            out = tf.layers.dense(hidden, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='a2', trainable=trainable)
            return tf.multiply(out, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            n_l2 = 60
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable, initializer=init_w)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable, initializer=init_w)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable, initializer=init_b)
            #  net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            hidden = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            hidden = tf.layers.dense(hidden, n_l2, activation=tf.nn.relu, kernel_initializer=init_w,
                                     bias_initializer=init_b, trainable=trainable, name='net2')
            out = tf.layers.dense(hidden, 1, trainable=trainable, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='net3')  # Q(s,a)
            return out
