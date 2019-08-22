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
ALGORITHM = 'maddpg'
DEBUG = False
RETRAIN = True
IMITATION_EPISODE = 100000
VAR = 4  # control exploration
DECAY = .99999
MIN_VAR = 0.02


def update():
    if ALGORITHM == 'maddpg':
        ddpg = MADDPG(avs.n_actions, avs.n_features, 1, 'maddpg model', RETRAIN)
    elif ALGORITHM == 'ddpg':
        ddpg = DDPG(avs.n_actions, avs.n_features, 1, 'ddpg model', RETRAIN)
    else:
        ddpg = DDPG(avs.n_actions, avs.n_features, 1, 'ddpg model', RETRAIN)
    t1 = time.time()
    rewards1 = 0
    rewards2 = 0
    var = VAR
    collision = 0
    avgreward1 = []
    avgreward2 = []
    collision_percentage = []
    for i in range(MAX_EPISODES):
        s1, s2 = avs.reset()
        ep_reward1 = 0
        ep_reward2 = 0
        if i % 100000 == 0 and i > IMITATION_EPISODE:
            plot(avgreward1, avgreward2, collision_percentage, i)
        for j in range(MAX_EP_STEPS):
            if RENDER:
                avs.render()

            # Add exploration noise
            if i < IMITATION_EPISODE or i % 4 == 0:
                a1 = imitation(avs.agent1, avs.agent2, avs.target1)
                a2 = imitation(avs.agent2, avs.agent1, avs.target2)
            else:
                # add randomness to action selection for exploration
                a1 = ddpg.choose_action(s1)
                a1 = [np.clip(np.random.normal(a1[0], var), -1, 1), np.clip(np.random.normal(a1[1], var), -1, 1)]
                a2 = ddpg.choose_action(s2)
                a2 = [np.clip(np.random.normal(a2[0], var), -1, 1), np.clip(np.random.normal(a2[1], var), -1, 1)]
                # a2 = imitation(avs.agent2, avs.agent1, avs.target2)

            if DEBUG:
                time.sleep(0.1)
            s_1, r1, s_2, r2, done, info = avs.step(a1, a2)
            if ALGORITHM == 'ddpg':
                ddpg.store_transition(s1, a1, r1, s_1)
                ddpg.store_transition(s2, a2, r2, s_2)
            else:
                ddpg.store_transition(s1, s2, a1, a2, r1, s_1, s_2)
                ddpg.store_transition(s2, s1, a2, a1, r2, s_2, s_1)

            s1 = s_1
            s2 = s_2
            ep_reward1 += r1
            ep_reward2 += r2

            if j == MAX_EP_STEPS - 1 or done:
                print("pt:", ddpg.pointer)
                print('Episode:', i, 'Step:', j, ' Reward: %i' % int(ep_reward1), int(ep_reward2),
                      'Explore: %.2f' % var)

                if i >= IMITATION_EPISODE:
                    rewards1 += ep_reward1
                    rewards2 += ep_reward2
                    if r1 < -100:
                        collision += 1
                    if (i + 1) % 100 == 0:
                        avgreward1.append(rewards1 / 100)
                        avgreward2.append(rewards2 / 100)
                        collision_percentage.append(collision)
                        rewards1 = 0
                        rewards2 = 0
                        collision = 0
                break
        if ddpg.pointer > MEMORY_CAPACITY:
            ddpg.learn()
            ddpg.learn()
            if var > MIN_VAR and i > IMITATION_EPISODE:
                var *= DECAY  # decay the action randomness
        if i % 4 != 0 and ep_reward1 > 100 and ep_reward2 > 100 and i > IMITATION_EPISODE:
            ddpg.save(i)
    print('Running time: ', time.time() - t1)


def plot(avgreward1, avgreward2, collision_percentage, i):
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率
    t = np.arange(0, len(avgreward1), 1)
    plt.subplot(3, 1, 1)
    plt.plot(t, avgreward1, 'r', t, avgreward2, 'b')
    label = ['Agent1', 'Agent2']
    plt.legend(label, loc='upper left')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-210, 200)
    # plt.savefig('plot/'+str(self.pointer)+'.jpg')
    t = np.arange(0, len(collision_percentage), 1)
    plt.subplot(3, 1, 3)
    plt.plot(t, collision_percentage, 'b')
    plt.xlabel('Episode')
    plt.ylabel('collision_percentage %')
    plt.ylim(0, 100)
    plt.savefig(ALGORITHM + ' plot/' + str(i) + '.jpg')


def imitation(ag1, ag2, tg1):
    action = [0, 0]
    if -11 < ag1.x - tg1.x <= 4 and abs(ag1.y - tg1.y) < 2.2 and pi / 20 > ag1.delta >= 0 and ag1.x - ag2.x > 0:
        action[0] = 1
        if ag1.y - ag2.y > 0:
            action[1] = -1
        else:
            action[1] = 1
    elif 11 > tg1.x - ag1.x >= -4 and abs(ag1.y - tg1.y) < 2.2 and pi / 20 > ag1.delta >= 0 and \
            ag1.x - ag2.x < 0 and (abs(ag1.y - ag2.y) > 4 or abs(ag1.x - ag2.x) > 5):
        action[0] = 1
        if ag1.y - ag2.y > 0:
            action[1] = -1
        else:
            action[1] = 1
    elif 11 > tg1.x - ag1.x >= -4 and abs(ag1.y - tg1.y) < 2.2 and pi / 20 > ag1.delta >= 0 and \
            ag1.x - ag2.x < 0 and abs(ag2.y - ag1.y) < 4 and abs(ag2.x - ag1.x) < 5:
        action[0] = -1
        action[1] = 0
    elif -11 < ag1.x - tg1.x <= 4 and ag1.delta == pi / 20 and abs(ag1.y - tg1.y) < 1.3:
        action[0] = 0
        action[1] = 0
        # print(2)
    elif -11 < ag1.x - tg1.x <= 4 and abs(ag1.y - tg1.y) > 1.8 and ag1.delta > 0:
        action[0] = 1
        if ag1.y - ag2.y > 0:
            action[1] = 1
        else:
            action[1] = -1
    elif -11 < ag1.x - tg1.x <= 4 and abs(ag1.y - tg1.y) > 1.8 and ag1.delta == 0:
        action[0] = 0
        action[1] = 0
    elif ag1.x - tg1.x > 4 and abs(ag1.y - tg1.y) > 1.5 and ag1.delta == 0:
        action[0] = 0
        if ag1.y - ag2.y > 0:
            action[1] = 1
        else:
            action[1] = -1
        # print(5)
    elif ag1.x - tg1.x > 0 and 0 < abs(ag1.y - tg1.y) < 1 and abs(ag1.delta - 39 * pi / 20) < 0.1:
        action[0] = 0
        if ag1.y - ag2.y > 0:
            action[1] = -1
        else:
            action[1] = 1
        # print(6)
    elif ag1.x - tg1.x > 4 and abs(ag1.y - tg1.y) < 1 and ag1.delta == 0:
        action[0] = 1
        action[1] = 0
        # print(7)
    elif ag1.x - tg1.x <= -11:
        action[0] = 0
        action[1] = 0
        # print(8)
    else:
        action[0] = 0
        action[1] = 0
    return action


if __name__ == "__main__":
    avs = AVS()
    avs.after(100, update)
    avs.mainloop()
    if OUTPUT_GRAPH:
        print('神经网络的日志文件生成在 logs 文件夹里，请用以下命令在 TensorBoard 中查看网络模型图：')
        print('\ttensorboard --logdir=logs\n')
