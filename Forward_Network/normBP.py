#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-15 13:12:06
# @Author  : blhoy
# @email   : 2351182903@qq.com

import random
import sys
import math
import codecs as cs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

#linear
class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1
#sigmoid
class SigmoidActivator(object):
    def forward(self, weight_input):
        return 1.0 / (1.0 + np.exp(-weight_input))

    def backward(self, output):
        return output * (1 - output)
#relu
class ReluActivator(object):
    def forward(self, weighted_input):
        return weighted_input * (weighted_input > 0)

    def backward(self, output):
        return 1 * (output > 0)
#tanh
class TanhActivator(object):
    def forward(self, weighted_input):
        return 2.0 / (1.0 + np.exp(-2 * weighted_input)) - 1.0

    def backward(self, output):
        return 1 - output * output

class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator = 'sigmoid'):
        self.input_size = input_size
        self.output_size = output_size
        if activator == 'sigmoid':
            std = np.sqrt(2.0/( (input_size+output_size) ))
            self.W = np.random.normal(0., std, (output_size, input_size))
            self.activator = SigmoidActivator()
        elif activator == 'identity':
            self.activator = IdentityActivator()
            std = np.sqrt(2.0/( (input_size+output_size) ))
            self.W = np.random.normal(0., std, (output_size, input_size))
        elif activator == 'relu':
            self.activator = ReluActivator()
            std = np.sqrt(2.0) * np.sqrt(2.0/( (input_size+output_size) ))
            self.W = np.random.normal(0., std, (output_size, input_size))
        elif activator == 'tanh':
            self.activator = TanhActivator()
            # ranges = np.sqrt(3)*0.01
            # self.W = np.random.uniform(-ranges, ranges,
            # (output_size, input_size))
            std = np.sqrt(2.0/( (input_size+output_size) ))
            self.W = np.random.normal(0., std, (output_size, input_size))
        else:
            self.activator = ReluActivator()
            std = np.sqrt(2.0) * np.sqrt(2.0/( (input_size+output_size) ))
            self.W = np.random.normal(0., std, (output_size, input_size))

        self.b = np.zeros((output_size, 1))
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        self.input = input_array
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        self.delta = self.activator.backward(self.input) * np.dot(
            self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print 'W: %s\nb:%s' % (self.W, self.b)

class Network(object):
    def __init__(self, layers, activator='sigmoid', ac_last = 'sigmoid', learning_rate=0.01, epoch=1000, loss='LMS', KFold=5):
        self.learning_rate = learning_rate
        self.layers = []
        self.loss = loss
        self.epoch = epoch
        self.KFold = KFold
        tmp = 0
        for i in xrange(len(layers) - 2):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    activator
                    )
                )
            tmp = i+1
        self.layers.append(
                FullConnectedLayer(
                    layers[tmp], layers[tmp+1],
                    ac_last
                    )
                )
    def predict(self, sample):
        res = sample
        for layer in self.layers:
            layer.forward(res)
            res = layer.output
        return res

    def get_loss(self, output, label):
        if self.loss == 'LMS':
            return 0.5 * ((label - output) * (label - output)).sum()
        else:
            tmp = 1.0 - output + 1e-8
            return -1.0 * (label * math.log(output, 2) + (1.0 - label) * math.log(tmp, 2) ).sum()

    #针对二分类的结果判断
    def train_predict(self, data_set, labels):
        loop = 50
        kf=KFold(n_splits=self.KFold, shuffle=True)
        acc = .0
        total = .0
        test_len = 0
        train_loss = .0
        for i in range(self.epoch+1):
            if (i+1)%loop == 0:
                acc = .0
                total = .0
                test_len = 0
                print 'epoch %d on...'%(i+1)
            for train_index,test_index in kf.split(data_set):
                # print("Train Index:",train_index,",Test Index:",test_index)
                X_train,X_test = data_set[train_index], data_set[test_index]
                y_train,y_test = labels[train_index], labels[test_index]
                test_len = len(y_test)
                for d in xrange(len(X_train)):
                    self.train_one_sample(y_train[d],
                        X_train[d])
                correct = .0
                for sample, y in zip(X_test, y_test):
                    res = self.predict(sample)
                    train_loss += self.get_loss(res, y)
                    if(math.fabs(y - res) < 0.5):
                        correct += 1
                acc += correct
                total += test_len
            if (i+1)%loop == 0:
                print 'the train_loss: ',train_loss / (self.KFold*loop)
                train_loss = .0
                print 'the accurancy is: ',acc/total * 100,'%'

    def train_one_sample(self, label, sample):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight()

    def calc_gradient(self, label):
        if self.loss == 'LMS':
            delta = self.layers[-1].activator.backward(
                self.layers[-1].output
            ) * (label - self.layers[-1].output)
        elif self.loss == 'CE':
            #binary cross entropy
            mid = (self.layers[-1].output)*(1. - self.layers[-1].output)
            if mid < 1E-14:
                mid = 1E-14
            delta = self.layers[-1].activator.backward(
                self.layers[-1].output
            ) * (label - self.layers[-1].output) / mid
        else:
            delta = self.layers[-1].activator.backward(
                self.layers[-1].output
            ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self):
        for layer in self.layers:
            layer.update(self.learning_rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

def read_data(path):
    x = []
    fx = cs.open(path +'\\' + 'ex4x.dat', 'r')
    for line in fx.readlines():
        tmp = np.array([.0,.0]).reshape((2,1))
        line = line.strip()
        tmp[0] = float(line.split()[0])
        tmp[1] = float(line.split()[1])
        x.append(tmp)

    y = []
    fy = cs.open(path +'\\' + 'ex4y.dat', 'r')
    for line in fy.readlines():
        line = line.strip()
        y.append(float(line))

    fx.close()
    fy.close()
    x = np.array(x)
    y = np.array(y)
    return x, y

if __name__ == '__main__':
    layerss = [2,10,1]
    acti='relu'
    learn=0.01
    epo=800
    cost = 'LMS'
    lenth = len(sys.argv)
    for i in range(1,lenth,2):
        if sys.argv[i] == '-L':
            lay = []
            for item in sys.argv[i+1].split(','):
                lay.append(int(item))
            layerss = lay
        if sys.argv[i] == '-a':
            acti = sys.argv[i+1]
        if sys.argv[i] == '-l':
            learn = float(sys.argv[i+1])
        if sys.argv[i] == '-e':
            epo = int(sys.argv[i+1])
        if sys.argv[i] == '-c':
            cost = sys.argv[i+1]

    dir_path = u"E:\study_and_learn\研一上\机器学习"
    x, y = read_data(dir_path)
    x /= np.max(x)
    
    net = Network(layers = layerss ,activator=acti ,learning_rate=learn ,epoch=epo, loss=cost )
    # , activator='relu'
    # net.dump()
    net.train_predict(x, y)
    # net.dump()
    # for xx, yy in zip(x, y):
    #     print net.predict(xx),' ',yy
    
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.xlabel('x1')
    plt.ylabel('x0')
    for xx, yy in zip(x, y):
        if (yy == 0):
            ax.plot(xx[0],xx[1],'ob' )
        else:
            ax.plot(xx[0],xx[1],'og')
    xn = np.linspace(0.,1.,50)
    yn = np.linspace(0.4,1.,50)
    res = []
    for xx0 in xn:
        vec = []
        for xx1 in yn:
            tmp = np.array([xx0,xx1]).reshape((2,1))
            if net.predict(tmp)<0.5:
                vec.append(0)
            else:
                vec.append(1)
        res.append(vec)
    # print res
    finx = []
    finy = []
    for i in xrange(len(res)):
        for j in xrange(1,len(res[i])):
            if res[i][j] == 1 and res[i][j-1] == 0:
                # ax.plot(xn[i],yn[j],'+r')
                # ax.plot(xn[i],yn[j-1],'+r')
                finx.append(xn[i])
                # finx.append(xn[i])
                # finy.append(yn[j-1])
                finy.append(yn[j])
    plt.plot(finx, finy,'-r')
    plt.title('Muitl_layer Network')
    plt.show()
    