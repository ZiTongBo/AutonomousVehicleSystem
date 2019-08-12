#!/usr/bin/avs python

# -*- coding: utf-8 -*-

# author：Elan time:2019/8/9

import sys
from env import *
from ddpg import *
import time
import numpy as np
import matplotlib.pyplot as plt
from maddpg import MADDPG
from math import *

MAX_EPISODES = 2000000
MAX_EP_STEPS = 2000
OUTPUT_GRAPH = True
RENDER = True
algorithm = 'ddpg'
DEBUG = False
RETRAIN = True
IMITATION_EPISODE = 1000


def update_ddpg():
    ddpg = DDPG(avs.n_actions, avs.n_features, 1, 'ddpg model', RETRAIN)
    t1 = time.time()
    rewards = 0
    cl = 0
    var = 3  # 方差 探索率
    avgreward = []
    collision = []
    # RENDER=False

    for i in range(MAX_EPISODES):
        s1, _ = avs.reset()
        ep_reward = 0
        if i % 5000 == 0 and i > IMITATION_EPISODE:
            plot(avgreward, collision, i)
            ddpg.save(i)
        for j in range(MAX_EP_STEPS):
            if RENDER:
                avs.render()

            # 模仿学习
            if i < IMITATION_EPISODE or i % 4 == 0:
                a1 = imitation(avs.agent1, avs.agent2, avs.target1)
            else:
                # add randomness to action selection for exploration
                a1 = ddpg.choose_action(s1)
                a1 = [np.clip(np.random.normal(a1[0], var), -1, 1), np.clip(np.random.normal(a1[1], var), -1, 1)]

            a2 = [0, 0]
            s_, r, _, _, done, info = avs.step(a1, a2)
            if DEBUG:
                time.sleep(0.02)

            ddpg.store_transition(s1, a1, r, s_)
            # print("pt:",ddpg.pointer)
            if ddpg.pointer > MEMORY_CAPACITY:
                if var > 0.01:
                    var *= .99998  # decay the action randomness
                ddpg.learn()

            s1 = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1 or done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, 'Step', j)
                print("pt:", ddpg.pointer)
                if i % 100 == 0:
                    rewards += ep_reward
                    if i >= IMITATION_EPISODE:
                        avgreward.append(rewards / 100)
                    rewards = 0
                else:
                    rewards += ep_reward
                if r < -100:
                    cl += 1
                if (i + 1) % 100 == 0:
                    if i >= IMITATION_EPISODE:
                        collision.append(cl)
                    cl = 0
                # if ep_reward > -300:RENDER = True
                break
    print('Running time: ', time.time() - t1)


def update_maddpg():
    var = 1  # control exploration
    # ddpg = MADDPG(avs.n_actions,avs.n_features,1,'nmodel')
    maddpg = MADDPG(avs.n_actions, avs.n_features, 1, 'mamodel', RETRAIN)
    t1 = time.time()
    rewards1 = 0
    rewards2 = 0
    cl = 0
    avgreward1 = []
    avgreward2 = []
    collision = []
    for i in range(MAX_EPISODES):
        s1, s2 = avs.reset1()
        ep_reward1 = 0
        ep_reward2 = 0

        for j in range(MAX_EP_STEPS):
            if RENDER:
                avs.render()

            # Add exploration noise
            # print(s1)
            a1 = maddpg.choose_action(s1)
            a2 = [0, 0]
            # print(a1)

            a1 = [np.clip(np.random.normal(a1[0], var), -1, 1), np.clip(np.random.normal(a1[1], var), -1, 1)]
            # a2 = [np.clip(np.random.normal(a2[0], var), -1, 1),np.clip(np.random.normal(a2[1], var), -1, 1)]
            # avs.agent1.v=15
            # a1=[0,0]
            ag1 = avs.agent1
            ag2 = avs.agent2
            tg1 = avs.target1
            tg2 = avs.target2
            if i < 1000 or (i > 1000 and i % 8 == 0):
                if 10 > tg1.x - ag1.x >= -4 and -1 <= ag1.y - tg1.y < 2.2 and pi / 20 > ag1.delta >= 0 and \
                        ag1.x - ag2.x > 0:
                    a1[0] = 1
                    a1[1] = 1
                    # print(1)
                elif 10 > tg1.x - ag1.x >= -4 and -1 <= ag1.y - tg1.y < 2.2 and pi / 20 > ag1.delta >= 0 and \
                        (abs(ag1.y - ag2.y) > 3 or abs(ag1.x - ag2.x) > 5):
                    a1[0] = 1
                    a1[1] = 1
                    # print(1)
                elif 10 > tg1.x - ag1.x >= -4 and -1 <= ag1.y - tg1.y < 2.2 and pi / 20 > ag1.delta >= 0 and \
                        abs(ag1.y - ag2.y) < 3 and abs(ag1.x - ag2.x) < 5:
                    a1[0] = -1
                    a1[1] = 0
                elif tg1.x - ag1.x < 10 and ag1.x - tg1.x <= 4 and ag1.delta >= pi / 20 and ag1.y - tg1.y < 1.3:
                    a1[0] = 0
                    a1[1] = 0
                    # print(2)
                elif tg1.x - ag1.x < 10 and ag1.x - tg1.x <= 4 and ag1.y - tg1.y > 1.8 and ag1.delta > 0:
                    a1[1] = -1
                    a1[0] = 1
                    # print(3)
                elif tg1.x - ag1.x < 10 and ag1.x - tg1.x <= 4 and ag1.y - tg1.y > 1.8 and ag1.delta == 0:
                    a1[1] = 0
                    a1[0] = 0
                    # print(4)
                elif ag1.x - tg1.x > 4 and ag1.y - tg1.y > 1.5 and ag1.delta == 0:
                    a1[1] = -1
                    a1[0] = 0
                    # print(5)
                elif ag1.x - tg1.x > 0 and 0 < ag1.y - tg1.y < 1 and abs(ag1.delta - 39 * pi / 20) < 0.1:
                    a1[1] = 1
                    a1[0] = 0
                    # print(6)
                elif ag1.x - tg1.x > 4 and -1 < ag1.y - tg1.y < 1 and ag1.delta == 0:
                    a1[1] = 0
                    a1[0] = 1
                    # print(7) 
                elif tg1.x - ag1.x >= 10:
                    a1[0] = 0.5
                    a1[1] = 0
                    # print(8)
                else:
                    a1[0] = 0
                    a1[1] = 0
            if i < 200000 or (i > 3000 and i % 8 == 0):
                if 10 > tg2.x - ag2.x >= -4 and -1 <= ag2.y - tg2.y < 2.2 and pi / 20 > ag2.delta and ag2.x - ag1.x > 0:
                    a2[0] = 1
                    a2[1] = 1
                    # print(1)
                elif 10 > tg2.x - ag2.x >= -4 and -1 <= ag2.y - tg2.y < 2.2 and pi / 20 > ag2.delta >= 0 and \
                        (abs(ag2.y - ag1.y) > 3 or abs(ag2.x - ag1.x) > 5):
                    a2[0] = 1
                    a2[1] = 1
                elif 10 > tg2.x - ag2.x >= -4 and -1 <= ag2.y - tg2.y < 2.2 and ag2.delta / 20 and ag2.delta >= 0 and \
                        abs(ag2.y - ag1.y) < 3 and abs(ag2.x - ag1.x) < 5:
                    a2[0] = -1
                    a2[1] = 0
                elif tg2.x - ag2.x < 10 and ag2.x - tg2.x <= 4 and ag2.delta == pi / 20 and ag2.y - tg2.y < 1.3:
                    a2[0] = 0
                    a2[1] = 0
                    # print(2)
                elif tg2.x - ag2.x < 10 and ag2.x - tg2.x <= 4 and ag2.y - tg2.y > 1.8 and ag2.delta > 0:
                    a2[1] = -1
                    a2[0] = 1
                    # print(3)
                elif tg2.x - ag2.x < 10 and ag2.x - tg2.x <= 4 and ag2.y - tg2.y > 1.8 and ag2.delta == 0:
                    a2[1] = 0
                    a2[0] = 0
                    # print(4)
                elif ag2.x - tg2.x > 4 and ag2.y - tg2.y > 1.5 and ag2.delta == 0:
                    a2[1] = -1
                    a2[0] = 0
                    # print(5)
                elif ag2.x - tg2.x > 0 and 0 < ag2.y - tg2.y < 1 and abs(ag2.delta - 39 * pi / 20) < 0.01:
                    a2[1] = 1
                    a2[0] = 0
                    # print(6)
                elif ag2.x - tg2.x > 4 and -1 < ag2.y - tg2.y < 1 and ag2.delta == 0:
                    a2[1] = 0
                    a2[0] = 1
                    # print(7) 
                elif tg2.x - ag2.x >= 10:
                    a2[0] = 0.5
                    a2[1] = 0
                    # print(8)
                else:
                    a2[0] = 0
                    a2[1] = 0
                    # print(9)
            # a2=[np.clip(np.random.normal(a2[0], 0.1), -1, 1),np.clip(np.random.normal(a2[1], 0.1), -1, 1)]
            if (i + 1) % 100 == 0:
                print(s1)
                print(a1)
                print(s2)
                print(a2)
            if DEBUG:
                time.sleep(0.02)
            s_1, r1, s_2, r2, done, info = avs.step1(a1, a2)

            maddpg.store_transition(s1, s2, a1, a2, r1, s_1, s_2)

            if maddpg.pointer > MEMORY_CAPACITY:
                maddpg.learn()
                if var > 0.02:
                    var *= .99997  # decay the action randomness
                # ddpg.learn()

            s1 = s_1
            s2 = s_2
            ep_reward1 += r1
            ep_reward2 += r2

            if j == MAX_EP_STEPS - 1 or done:
                # if ep_reward1 > -2000:
                # for m1 in mm1:
                # ddpg.store_transition(m1.s1, m1.a1,  m1.r , m1.s_1)

                print("pt:", maddpg.pointer)
                print('Episode:', i, 'Step:', j, ' Reward: %i' % int(ep_reward1), int(ep_reward2),
                      'Explore: %.2f' % var, )

                if i >= 1000:
                    rewards1 += ep_reward1
                    rewards2 += ep_reward2
                if (i + 1) % 100 == 0 and i >= 1000:
                    rewards1 += ep_reward1
                    rewards2 += ep_reward2
                    avgreward1.append(rewards1 / 100)
                    avgreward2.append(rewards2 / 100)
                    rewards1 = 0
                    rewards2 = 0

                # if ep_reward > -300:RENDER = True
                if r1 < -200 and i >= 1000:
                    cl += 1
                if (i + 1) % 100 == 0 and i >= 1000:
                    collision.append(cl)
                    cl = 0
                break
    print('Running time: ', time.time() - t1)


def plot(avgreward, collision, i):
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率
    t = np.arange(0, len(avgreward), 1)
    plt.subplot(3, 1, 1)
    plt.plot(t, avgreward, 'r', )
    label = ['Agent1', 'Agent2']
    plt.legend(label, loc='upper left')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-210, 200)
    # plt.savefig('plot/'+str(self.pointer)+'.jpg')
    t = np.arange(0, len(collision), 1)
    plt.subplot(3, 1, 3)
    plt.plot(t, collision, 'b')
    plt.xlabel('Episode')
    plt.ylabel('collision %')
    plt.ylim(0, 100)
    plt.savefig('plot1/' + str(i) + '.jpg')


def imitation(ag1, ag2, tg1):
    a1 = [0, 0]
    if -10 < ag1.x - tg1.x <= 4 and -1 <= ag1.y - tg1.y < 2.2 and pi / 20 > ag1.delta >= 0:
        a1[0] = 1
        a1[1] = 1
    elif -10 < ag1.x - tg1.x <= 4 and ag1.delta >= pi / 20 and ag1.y - tg1.y < 1.3:
        a1[0] = 0
        a1[1] = 0
        # print(2)
    elif -10 < ag1.x - tg1.x <= 4 and ag1.y - tg1.y > 1.8 and ag1.delta > 0:
        a1[1] = -1
        a1[0] = 1
        # print(3)
    elif -10 < ag1.x - tg1.x <= 4 and ag1.y - tg1.y > 1.8 and ag1.delta == 0:
        a1[1] = 0
        a1[0] = 0
        # print(4)
    elif ag1.x - tg1.x > 4 and ag1.y - tg1.y > 1.5 and ag1.delta == 0:
        a1[1] = -1
        a1[0] = 0
        # print(5)
    elif ag1.x - tg1.x > 0 and 0 < ag1.y - tg1.y < 1 and abs(ag1.delta - 39 * pi / 20) < 0.1:
        a1[1] = 1
        a1[0] = 0
        # print(6)
    elif ag1.x - tg1.x > 4 and -1 < ag1.y - tg1.y < 1 and ag1.delta == 0:
        a1[1] = 0
        a1[0] = 1
        # print(7)
    elif ag1.x - tg1.x <= -10:
        a1[0] = 0.5
        a1[1] = 0
        # print(8)
    else:
        a1[0] = 0
        a1[1] = 0
    return a1


if __name__ == "__main__":
    avs = AVS()
    if algorithm == 'ddpg':
        avs.after(100, update_ddpg)
        avs.mainloop()
    elif algorithm == 'maddpg':
        avs.after(100, update_maddpg)
        avs.mainloop()

    if OUTPUT_GRAPH:
        print('神经网络的日志文件生成在 logs 文件夹里，请用以下命令在 TensorBoard 中查看网络模型图：')
        print('\ttensorboard --logdir=logs\n')
