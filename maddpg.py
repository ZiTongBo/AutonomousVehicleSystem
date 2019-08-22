import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)
# tf.compat.v1.set_random_seed(1)

LR_A = 0.0005  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]  # you can try different target replacement strategies
MEMORY_CAPACITY = 10000
BATCH_SIZE = 256
RENDER = False
OUTPUT_GRAPH = True
TAU = 0.01


class MADDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, model, retrain):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 4 + a_dim * 2 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.model = model
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S1 = tf.placeholder(tf.float32, [None, s_dim], 's1')
        self.S_1 = tf.placeholder(tf.float32, [None, s_dim], 's_1')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        self.S2 = tf.placeholder(tf.float32, [None, s_dim], 's2')
        self.S_2 = tf.placeholder(tf.float32, [None, s_dim], 's_2')
        self.a2 = tf.placeholder(tf.float32, [None, a_dim], 'a2')
        self.retrain = retrain

        # self.saver = tf.train.import_meta_graph('./nmodel/model-45000.meta')
        # ckpt = tf.train.get_checkpoint_state('./nmodel')
        # if ckpt and ckpt.model_checkpoint_path and not self.retrain:
        #     self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        #     print(1)
        #     self.graph = tf.get_default_graph()
        with tf.variable_scope('Actor'):
            self.a1 = self._build_a(self.S1, scope='eval', trainable=True)
            a_1 = self._build_a(self.S_1, scope='target', trainable=False)
            a_2 = self._build_a(self.S_2, scope='target', trainable=False)
            # self.a2 = self._build_a(self.S2, scope='eval', trainable=True)
            # self.a_2 = self._build_a(self.S_2, scope='target', trainable=False)

        #  使用DDPG训练的Actor
        # self.ae_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        # self.at_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        # self.ae_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/eval')
        # self.at_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/target')
        # self.replace1 = [[tf.assign(ta, ea), tf.assign(tc, ec)]
        #                  for ta, ea, tc, ec in zip(self.ae_params2, self.ae_params1, self.at_params2,self.at_params1)]

        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S1, self.S2, self.a1, self.a2, scope='eval', trainable=True)
            q_ = self._build_c(self.S_1, self.S_2, a_1, a_2, scope='target', trainable=False)

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

        #
        self.sess.run(tf.global_variables_initializer())
        # self.sess.run(self.replace1)
        self.avgreward = []
        self.collision = []
        self.saver = tf.train.Saver(max_to_keep=4)
        # ckpt = tf.train.get_checkpoint_state('./nmodel')
        # self.saver = saver = tf.train.Saver(max_to_keep=4)
        # if ckpt and ckpt.model_checkpoint_path:
        # self.saver.restore(self.sess,ckpt.model_checkpoint_path)
        #    print(1)

    def choose_action(self, s):
        return self.sess.run(self.a1, {self.S1: s[np.newaxis, :]})[0]

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs1 = bt[:, :self.s_dim]
        bs2 = bt[:, self.s_dim: self.s_dim * 2]
        ba1 = bt[:, self.s_dim * 2: self.s_dim * 2 + self.a_dim]
        ba2 = bt[:, self.s_dim * 2 + self.a_dim: self.s_dim * 2 + self.a_dim * 2]
        br = bt[:, -self.s_dim * 2 - 1: -self.s_dim * 2]
        bs_1 = bt[:, -self.s_dim * 2: -self.s_dim]
        bs_2 = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S1: bs1, self.S2: bs2, self.a2: ba2})
        self.sess.run(self.ctrain,
                      {self.S1: bs1, self.S2: bs2, self.a1: ba1, self.a2: ba2, self.R: br, self.S_1: bs_1,
                       self.S_2: bs_2})

    def save(self, episode):
        self.saver.save(self.sess, self.model + '/' + self.model, global_step=episode)

    def store_transition(self, s1, s2, a1, a2, r, s_1, s_2):
        transition = np.hstack((s1, s2, a1, a2, [r], s_1, s_2))
        index = self.pointer % MEMORY_CAPACITY  # r  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            # net = self.graph.get_tensor_by_name('Actor/'+scope+'/l1/Tanh:0')
            # a1 = self.graph.get_tensor_by_name('Actor/'+scope+'/a1/Tanh:0')
            # a2 = self.graph.get_tensor_by_name('Actor/'+scope+'/a2/Tanh:0')
            # tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor1/eval')
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.tanh, kernel_initializer=init_w,
                                  bias_initializer=init_b, name='l11', trainable=trainable, reuse=tf.AUTO_REUSE)
            # a = tf.layers.dense(net, self.a_dim, name='a', trainable=trainable)
            a1 = tf.layers.dense(net, 50, activation=tf.nn.tanh, kernel_initializer=init_w,
                                 bias_initializer=init_b, name='a11', trainable=trainable, reuse=tf.AUTO_REUSE)
            a2 = tf.layers.dense(a1, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                 bias_initializer=init_b, name='a21', trainable=trainable, reuse=tf.AUTO_REUSE)
            return a2

    def _build_c(self, s1, s2, a1, a2, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            n_l2 = 40
            w_s1 = tf.get_variable('w1_s1', [self.s_dim, n_l1], trainable=trainable)
            w_s2 = tf.get_variable('w2_s2', [self.s_dim, n_l1], trainable=trainable)
            w_a1 = tf.get_variable('w1_a1', [self.a_dim, n_l1], trainable=trainable)
            w_a2 = tf.get_variable('w1_oa1', [self.a_dim, n_l1], trainable=trainable)
            b = tf.get_variable('b', [1, n_l1], trainable=trainable)
            net1 = tf.nn.tanh(tf.matmul(s1, w_s1) + tf.matmul(s2, w_s2) + tf.matmul(a1, w_a1) + tf.matmul(a2, w_a2) + b)
            net2 = tf.layers.dense(net1, n_l2, activation=tf.nn.tanh, trainable=trainable, name='net21')
            net3 = tf.layers.dense(net2, 1, trainable=trainable, name='net31')  # Q(s,a)
            return net3
