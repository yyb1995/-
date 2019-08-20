# -*- coding: utf-8 -*-
# @Time  : 2019-08-19 20:19
# @Author: YYB
# Implement original Perception machine.

from BaseModel import BaseModel
import numpy as np


class PerceptionMachine(BaseModel):
    def __init__(self, x, y):
        """
        Initialize.
        :param x: Input. Shape: (batch, dim)
        :param y: Output. Shape: (batch, 1)
        """
        super(PerceptionMachine, self).__init__()
        self.x = x
        self.y = y

        assert self.x.ndim == 2 and self.y.ndim == 2, 'Input and output should be 2-dim numpy array.'
        self.w = np.random.rand(x.shape[1], 1)
        self.b = np.zeros([x.shape[0], 1])
        self.step = 0.5
        self.activate_func = np.sign

    def model(self, x):
        return self.activate_func(np.dot(x, self.w) + self.b)

    def param(self):
        return self.w, self.b

    def train(self):
        """
        Update process.
        :return:
        """
        iter_time = 0
        while not np.all(self.model(self.x) == self.y):
            # Choose 1 sample from data to update w and b.
            wrong_classfied = np.argwhere(self.model(self.x) != self.y)
            updated = wrong_classfied[np.random.choice(wrong_classfied.shape[0], 1, replace=False)[0]]
            self.w = self.w + self.step * self.x[tuple(updated)] * self.y[tuple(updated)]
            self.b = self.b + self.step * self.y[tuple(updated)]
            iter_time += 1
            print('iter_time: {}\nwrong_classfied_sample_left: {}'.format(
                iter_time, np.sum(self.model(self.x) != self.y)))
            print('current w: {}\ncurrent b: {}\n'.format(*self.param()))

    def predict(self, x):
        return self.model(x)


if __name__ == '__main__':
    x_train = np.array([[3, 3], [4, 3], [1, 1]])
    y_train = np.array([[1], [1], [-1]])
    perception1 = PerceptionMachine(x_train, y_train)
    perception1.train()
    print(perception1.predict(np.array([3, 3]))[0, 0])
