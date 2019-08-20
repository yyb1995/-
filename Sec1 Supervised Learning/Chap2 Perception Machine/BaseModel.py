# -*- coding: utf-8 -*-
# @Time  : 2019-08-19 20:37
# @Author: YYB


class BaseModel(object):
    def __init__(self, **kwargs):
        self.x = None
        self.y = None

    def param(self, **kwargs):
        pass

    def model(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def predict(self, **kwargs):
        pass
