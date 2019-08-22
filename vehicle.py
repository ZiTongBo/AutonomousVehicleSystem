#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author：Elan time:2019/8/9

import numpy as np
from math import *

dt = 0.2
A = 1
Delta = pi / 18
length = 4.2
UNIT = 20
width = 1.8
angel = atan(width / length)
hypotenuse = sqrt((length * length + width * width) / 4)
lanes = 3  # 车道数


class Vehicle(object):
    def __init__(self, x, y, delta, v, a, psi, f_len=1, r_len=1.2):
        self.object = None
        self.x = x
        self.y = y
        self.psi = psi  # 车辆当前的偏航角，逆时针为正
        self.v = v
        self.f_len = f_len  # 前轮到重心的距离，1m
        self.r_len = r_len  # 后轮到重心的距离，
        self.a = a  # 加速度，油门控制
        self.delta = delta  # 前轮转向角，方向盘控制
        self.last_action = ''
        self.point = None
        self.set_coordinates()
        self.avgreward = []
        self.collision = []
        self.done = False

    def get_state(self):
        return self.x, self.y, self.psi, self.v, self.delta, self.a

    def update_state(self, action):
        self.a = action[0]
        self.delta = action[1]
        beta = atan((self.r_len / (self.r_len + self.f_len)) * tan(self.delta))
        self.x = self.x + self.v * cos(self.psi + beta) * dt
        self.y = self.y + self.v * sin(self.psi + beta) * dt
        self.psi = self.psi + (self.v / self.f_len) * sin(beta) * dt
        self.v = self.v + self.a * dt
        self.set_coordinates()
        return self.x, self.y, self.psi, self.v

    def update(self, action):
        self.v = action[0] + self.v
        self.delta = (action[1] * pi / 20 + self.delta + 2 * pi) % (2 * pi)
        self.x += self.v * cos(self.delta) * dt
        self.y += self.v * sin(self.delta) * dt
        self.set_coordinates()

    def set_coordinates(self):
        x1 = self.x + hypotenuse * cos(self.psi + angel)
        y1 = self.y + hypotenuse * sin(self.psi + angel)
        x2 = self.x + hypotenuse * cos(self.psi - angel)
        y2 = self.y + hypotenuse * sin(self.psi - angel)
        x3 = self.x - hypotenuse * cos(self.psi + angel)
        y3 = self.y - hypotenuse * sin(self.psi + angel)
        x4 = self.x - hypotenuse * cos(self.psi - angel)
        y4 = self.y - hypotenuse * sin(self.psi - angel)
        self.point = UNIT * np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    def get_four_points(self):
        return self.point[0][0], self.point[0][1], self.point[1][0], self.point[1][1], \
               self.point[2][0], self.point[2][1], self.point[3][0], self.point[3][1]
