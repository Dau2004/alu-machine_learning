#!/usr/bin/env python3
'''
    A class Neuron that defines a single neuron performing
    binary classification:
'''


import numpy as np


class Neuron:
    '''
        Class Neuron
    '''
    def __init__(self, nx):
        '''
            Constructor
        '''
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
            Getter
        '''
        return self.__W

    @property
    def b(self):
        '''
            Getter
        '''
        return self.__b

    @property
    def A(self):
        '''
            Getter
        '''
        return self.__A

    def forward_prop(self, X):
        '''
            Calculates the forward propagation of the neuron
        '''
        self.__A = 1 / (1 + np.exp(-np.dot(self.__W, X) - self.__b))
        return self.__A

    def cost(self, Y, A):
        '''
            Calculates the cost of the model using logistic regression
        '''
        m = Y.shape[1]
        cost = ((-1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                  * np.log(1.0000001 - A)))
        return cost