#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2019/8/9

import sys
import numpy as np
import tkinter as tk
from math import *
from vehicle import Vehicle
import time

np.random.seed(0)
LENGTH = 4.2
WIDTH = 1.8
ANGEL = atan(WIDTH / LENGTH)
HYPOTENUSE = sqrt((LENGTH * LENGTH + WIDTH * WIDTH) / 4)
LANES = 3
LANE_WIDTH = 3.5
LANE_LENGTH = 100
MAX_VELOCITY = 30
UNIT = 20  # 1m=20px
SYMMETRY = True


class AVS(tk.Tk, object):
    def __init__(self):
        super(AVS, self).__init__()
        # self.action_space = spaces.Box(np.array([-1,-1]),np.array([1,1]))
        self.n_actions = 2
        self.n_features = 19
        self.title('AVS')
        self.geometry('{0}x{1}'.format(LANE_LENGTH * UNIT, int(LANE_WIDTH * UNIT * LANES)))
        self.canvas = tk.Canvas(self, bg='white',
                                width=LANE_LENGTH * UNIT, height=70 * LANES)
        mid = np.array([LENGTH / 2, LANE_WIDTH * LANES / 2])
        for c in range(70, 70 * LANES, 70):
            x0, y0, x1, y1 = 0, c, LANE_LENGTH * UNIT, c
            self.canvas.create_line(x0, y0, x1, y1)

        agent1_center = mid + np.array([0, -LANE_WIDTH])
        self.agent1 = Vehicle(x=agent1_center[0], y=agent1_center[1], psi=0, v=30, a=0, delta=0)
        self.agent1.object = self.canvas.create_polygon(self.agent1.get_four_points(), fill='blue')

        agent2_center = mid + np.array([0, LANE_WIDTH])
        self.agent2 = Vehicle(agent2_center[0], agent2_center[1], 0, 10, 0, 0)
        self.agent2.object = self.canvas.create_polygon(self.agent2.get_four_points(), fill='green')

        target1_center = mid + np.array([np.random.randint(25, 30), LANE_WIDTH])
        self.target1 = Vehicle(target1_center[0], target1_center[1], 0, 10, 0, 0)
        self.target1.object = self.canvas.create_polygon(self.target1.get_four_points(), fill='black')

        target2_center = mid + np.array([np.random.randint(20, 25), -LANE_WIDTH])
        self.target2 = Vehicle(target2_center[0], target2_center[1], 0, 10, 0, 0)
        self.target2.object = self.canvas.create_polygon(self.target2.get_four_points(), fill='black')

        target3_center = mid + np.array([np.random.randint(5, 15), 0])
        self.target3 = Vehicle(target3_center[0], target3_center[1], 0, 10, 0, 0)
        self.target3.object = self.canvas.create_polygon(self.target3.get_four_points(), fill='black')

        self.canvas.pack()

    # return s1, s2
    def reset(self):
        self.update()
        self.delete_all()
        mid = np.array([LENGTH / 2, LANE_WIDTH * LANES / 2])
        self.agent1.done = False
        self.agent2.done = False
        agent1_center = mid + np.array([3, -LANE_WIDTH])
        self.agent1 = Vehicle(agent1_center[0], agent1_center[1], psi=0, v=17, a=0, delta=0)
        self.agent1.object = self.canvas.create_polygon(self.agent1.get_four_points(), fill='blue')

        agent2_center = mid + np.array([0, LANE_WIDTH])
        self.agent2 = Vehicle(agent2_center[0], agent2_center[1], 0, 15, 0, 0)
        self.agent2.object = self.canvas.create_polygon(self.agent2.get_four_points(), fill='green')

        target1_center = mid + np.array([15, -LANE_WIDTH])
        self.target1 = Vehicle(target1_center[0], target1_center[1], 0, 12, 0, 0)
        self.target1.object = self.canvas.create_polygon(self.target1.get_four_points(), fill='black')

        target2_center = mid + np.array([15, LANE_WIDTH])
        self.target2 = Vehicle(target2_center[0], target2_center[1], 0, 12, 0, 0)
        self.target2.object = self.canvas.create_polygon(self.target2.get_four_points(), fill='black')

        target3_center = mid + np.array([2, 0])
        self.target3 = Vehicle(target3_center[0], target3_center[1], 0, 12, 0, 0)
        self.target3.object = self.canvas.create_polygon(self.target3.get_four_points(), fill='black')
        # self.canvas.create_rectangle(0, 0, LANE_LENGTH * UNIT, 0.2 * UNIT,fill = 'red')

        s1 = self.observation(self.agent1)
        s2 = self.observation(self.agent2)
        return s1, s2

    # return s1,reward1,done
    def step(self, action1, action2):
        self.agent1.update(action1)
        self.agent2.update(action2)
        self.target1.update([0, 0])
        self.target2.update([0, 0])
        self.target3.update([0, 0])
        self.delete_all()
        self.agent1.object = self.canvas.create_polygon(self.agent1.get_four_points(), fill='blue')
        self.agent2.object = self.canvas.create_polygon(self.agent2.get_four_points(), fill='green')
        self.target1.object = self.canvas.create_polygon(self.target1.get_four_points(), fill='black')
        self.target2.object = self.canvas.create_polygon(self.target2.get_four_points(), fill='black')
        self.target3.object = self.canvas.create_polygon(self.target3.get_four_points(), fill='black')
        s1 = self.observation(self.agent1)
        s2 = self.observation(self.agent2)
        reward1 = 0
        reward2 = 0
        done = False
        info = None
        if self.agent1.x - self.target1.x > 6 and abs(self.agent1.y - self.target1.y) < 1:  # 成功奖励
            if not self.agent1.done:
                reward1 += 200
                self.agent1.done = True
            # done = True
            if self.agent2.done:
                done = True
        if self.agent2.x - self.target2.x > 6 and abs(self.agent2.y - self.target2.y) < 1:
            if not self.agent2.done:
                reward2 += 200
                self.agent2.done = True
            if self.agent1.done:
                done = True
        if self.collision(self.agent1):  # 碰撞惩罚
            reward1 -= 200
            done = True
        if (self.target1.x - self.agent1.x > 40 or self.target1.x > 100) and not self.agent1.done:  # 错失惩罚
            reward1 -= 50
            done = True
        if self.collision(self.agent2):
            reward2 -= 200
            done = True
        if (self.target2.x - self.agent2.x > 40 or self.target2.x > 100) and not self.agent2.done:
            reward2 -= 50
            done = True
        # reward1 += 1
        # reward2 += 1
        reward1 -= min(abs(self.agent1.delta), abs(2 * pi - self.agent1.delta)) / 2  # 转向惩罚
        reward2 -= min(abs(self.agent2.delta), abs(2 * pi - self.agent2.delta)) / 2
        reward1 -= min(abs(tanh(self.agent1.y - LANE_WIDTH / 2)), abs(tanh(self.agent1.y - LANE_WIDTH * 3 / 2)),
                       abs(tanh(self.agent1.y - LANE_WIDTH * 5 / 2)))  # 偏离惩罚
        reward2 -= min(abs(tanh(self.agent2.y - LANE_WIDTH / 2)), abs(tanh(self.agent2.y - LANE_WIDTH * 3 / 2)),
                       abs(tanh(self.agent2.y - LANE_WIDTH * 5 / 2)))
        return s1, reward1, s2, reward2, done, info

    def delete_all(self):
        self.canvas.delete(self.agent1.object)
        self.canvas.delete(self.agent2.object)
        self.canvas.delete(self.target1.object)
        self.canvas.delete(self.target2.object)
        self.canvas.delete(self.target3.object)

    def collision(self, vehicle):
        if len(self.canvas.find_overlapping(vehicle.point[0][0] - 1, vehicle.point[0][1] - 1, vehicle.point[0][0] + 1,
                                            vehicle.point[0][1] + 1)) > 1:
            if self.canvas.find_overlapping(vehicle.point[0][0] - 1, vehicle.point[0][1] - 1, vehicle.point[0][0] + 1,
                                            vehicle.point[0][1] + 1)[0] != 1 and \
                    self.canvas.find_overlapping(vehicle.point[0][0] - 1, vehicle.point[0][1] - 1,
                                                 vehicle.point[0][0] + 1, vehicle.point[0][1] + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[0][0]-1,vehicle.point[0][1]-1,vehicle.point[0][0]+1,vehicle.point[0][1]+1))
                return True
        if len(self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 2 - 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 2 - 1,
                                            (vehicle.point[0][0] + vehicle.point[1][0]) / 2 + 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 2 + 1)) > 1:
            if self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 2 - 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 2 - 1,
                                            (vehicle.point[0][0] + vehicle.point[1][0]) / 2 + 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 2 + 1)[0] != 1 and \
                    self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 2 - 1,
                                                 (vehicle.point[0][1] + vehicle.point[1][1]) / 2 - 1,
                                                 (vehicle.point[0][0] + vehicle.point[1][0]) / 2 + 1,
                                                 (vehicle.point[0][1] + vehicle.point[1][1]) / 2 + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[0][0]-1,vehicle.point[0][1]-1,vehicle.point[0][0]+1,vehicle.point[0][1]+1))
                return True
        if len(self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 4 - 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 4 - 1,
                                            (vehicle.point[0][0] + vehicle.point[1][0]) / 4 + 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 4 + 1)) > 1:
            if self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 4 - 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 4 - 1,
                                            (vehicle.point[0][0] + vehicle.point[1][0]) / 4 + 1,
                                            (vehicle.point[0][1] + vehicle.point[1][1]) / 4 + 1)[0] != 1 and \
                    self.canvas.find_overlapping((vehicle.point[0][0] + vehicle.point[1][0]) / 4 - 1,
                                                 (vehicle.point[0][1] + vehicle.point[1][1]) / 4 - 1,
                                                 (vehicle.point[0][0] + vehicle.point[1][0]) / 4 + 1,
                                                 (vehicle.point[0][1] + vehicle.point[1][1]) / 4 + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[0][0]-1,vehicle.point[0][1]-1,vehicle.point[0][0]+1,vehicle.point[0][1]+1))
                return True
        if len(self.canvas.find_overlapping(vehicle.point[1][0] - 1, vehicle.point[1][1] - 1, vehicle.point[1][0] + 1,
                                            vehicle.point[1][1] + 1)) > 1:
            if self.canvas.find_overlapping(vehicle.point[1][0] - 1, vehicle.point[1][1] - 1, vehicle.point[1][0] + 1,
                                            vehicle.point[1][1] + 1)[0] != 1 and \
                    self.canvas.find_overlapping(vehicle.point[1][0] - 1, vehicle.point[1][1] - 1,
                                                 vehicle.point[1][0] + 1, vehicle.point[1][1] + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[1][0]-1,vehicle.point[1][1]-1,vehicle.point[1][0]+1,vehicle.point[1][1]+1))
                return True
        if len(self.canvas.find_overlapping(vehicle.point[2][0] - 1, vehicle.point[2][1] - 1, vehicle.point[2][0] + 1,
                                            vehicle.point[2][1] + 1)) > 1:
            if self.canvas.find_overlapping(vehicle.point[2][0] - 1, vehicle.point[2][1] - 1, vehicle.point[2][0] + 1,
                                            vehicle.point[2][1] + 1)[0] != 1 and \
                    self.canvas.find_overlapping(vehicle.point[2][0] - 1, vehicle.point[2][1] - 1,
                                                 vehicle.point[2][0] + 1, vehicle.point[2][1] + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[2][0]-1,vehicle.point[2][1]-1,vehicle.point[2][0]+1,vehicle.point[2][1]+1))
                return True
        if len(self.canvas.find_overlapping(vehicle.point[3][0] - 1, vehicle.point[3][1] - 1, vehicle.point[3][0] + 1,
                                            vehicle.point[3][1] + 1)) > 1:
            if self.canvas.find_overlapping(vehicle.point[3][0] - 1, vehicle.point[3][1] - 1, vehicle.point[3][0] + 1,
                                            vehicle.point[3][1] + 1)[0] != 1 and \
                    self.canvas.find_overlapping(vehicle.point[3][0] - 1, vehicle.point[3][1] - 1,
                                                 vehicle.point[3][0] + 1, vehicle.point[3][1] + 1)[0] != 2:
                # print(self.canvas.find_overlapping(vehicle.point[3][0]-1,vehicle.point[3][1]-1,vehicle.point[3][0]+1,vehicle.point[3][1]+1))
                return True
        if vehicle.point[1][1] < 0 or vehicle.point[2][1] < 0:
            return True
        if vehicle.point[0][1] > LANE_WIDTH * UNIT * LANES or vehicle.point[3][1] > LANE_WIDTH * UNIT * LANES:
            return True
        return False

    def render(self):
        self.update()

    def observation(self, agent):
        if agent == self.agent1:
            # return np.array([self.agent1.x / LANE_LENGTH, self.agent1.y / (LANE_WIDTH * LANES),
            return np.array([self.agent1.y / (LANE_WIDTH * LANES),
                             (self.agent1.x - self.agent2.x) / LANE_LENGTH,
                             (self.agent1.y - self.agent2.y) / (LANE_WIDTH * LANES),
                             (self.agent1.x - self.target1.x) / LANE_LENGTH,
                             (self.agent1.y - self.target1.y) / (LANE_WIDTH * LANES),
                             (self.agent1.x - self.target2.x) / LANE_LENGTH,
                             (self.agent1.y - self.target2.y) / (LANE_WIDTH * LANES),
                             (self.agent1.x - self.target3.x) / LANE_LENGTH,
                             (self.agent1.y - self.target3.y) / (LANE_WIDTH * LANES),
                             self.agent1.v / MAX_VELOCITY, self.agent1.delta / (2 * pi),
                             self.agent2.v / MAX_VELOCITY, self.agent2.delta / (2 * pi),
                             self.target1.v / MAX_VELOCITY, self.target1.delta / (2 * pi),
                             self.target2.v / MAX_VELOCITY, self.target2.delta / (2 * pi),
                             self.target3.v / MAX_VELOCITY, self.target3.delta / (2 * pi)])
        if agent == self.agent2 and SYMMETRY:
            return np.array([self.agent2.y / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.agent1.x) / LANE_LENGTH,
                             (self.agent2.y - self.agent1.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target2.x) / LANE_LENGTH,
                             (self.agent2.y - self.target2.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target1.x) / LANE_LENGTH,
                             (self.agent2.y - self.target1.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target3.x) / LANE_LENGTH,
                             (self.agent2.y - self.target3.y) / (LANE_WIDTH * LANES),
                             self.agent2.v / MAX_VELOCITY, self.agent2.delta / (2 * pi),
                             self.agent1.v / MAX_VELOCITY, self.agent1.delta / (2 * pi),
                             self.target2.v / MAX_VELOCITY, self.target2.delta / (2 * pi),
                             self.target1.v / MAX_VELOCITY, self.target1.delta / (2 * pi),
                             self.target3.v / MAX_VELOCITY, self.target3.delta / (2 * pi)])
        else:
            return np.array([(LANE_WIDTH * LANES - self.agent2.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.agent1.x) / LANE_LENGTH,
                             (-self.agent2.y + self.agent1.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target2.x) / LANE_LENGTH,
                             (-self.agent2.y + self.target2.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target1.x) / LANE_LENGTH,
                             (-self.agent2.y + self.target1.y) / (LANE_WIDTH * LANES),
                             (self.agent2.x - self.target3.x) / LANE_LENGTH,
                             (-self.agent2.y + self.target3.y) / (LANE_WIDTH * LANES),
                             self.agent2.v / MAX_VELOCITY, (2 * pi - self.agent2.delta) % (2 * pi) / (2 * pi),
                             self.agent1.v / MAX_VELOCITY, (2 * pi - self.agent1.delta) % (2 * pi) / (2 * pi),
                             self.target2.v / MAX_VELOCITY, self.target2.delta / (2 * pi),
                             self.target1.v / MAX_VELOCITY, self.target1.delta / (2 * pi),
                             self.target3.v / MAX_VELOCITY, self.target3.delta / (2 * pi)])
