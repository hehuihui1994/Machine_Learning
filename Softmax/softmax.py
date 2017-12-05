# -*- coding: utf-8 -*-

import os
# import tools
import math,random
import pickle
import numpy as np


class SoftmaxClassifier(object):
    def __init__(self, feature_num, c = 2, estimator = 'sgd', penalty = 'l2', max_iter = 20):
        self.estimator = estimator
        self.feature_num = feature_num
        self.c = c
        self.penalty = penalty
        self.max_iter = max_iter
        self.theta = np.matrix([[0.]*(feature_num+1)] * c)    # C * feature_num+1

    def h(self, x):
        '''
        return : 1 * C vector if k==-1 ; true value if k in range(0, c)'''
        # if k==-1:
        lst = [math.exp(float(item)) for item in (x * np.transpose(self.theta)).tolist()[0]]
        _sum = sum(lst)
        vec = [item/_sum for item in lst]
        return vec

    def decison_function(self, X):
        '''
        h(x, self.theta) is a C * 1 vector
        mat =
        [ [ h1(x_1), h2(x_1), ..., hc(x_1) ],
          [ h1(x_2), h2(x_2), ..., hc(x_2) ],
          ...
          [ h1(x_n), h2(x_n), ..., hc(x_n) ]
        ]
        return mat's transpose( C * N ) i.e.
        [ [ h1(x_1), h1(x_2), ..., h1(x_n) ],
          [ h2(x_1), h2(x_2), ..., h2(x_n) ],
          ...
          [ hc(x_1), hc(x_2), ..., hc(x_n) ]
        ]
        '''
        mat = np.matrix([self.h(x) for x in X])
        return np.transpose(mat)

    def loss_function(self, decison_matrix, y):
        loss = 0.
        # i_th sample, label is k, corresponsing element hk(x_i)'s
        # position in decision_matrix is row k, col i
        for i, k in enumerate(y):
            hypo = decison_matrix.item( k*len(y) + i)
            loss += math.log(hypo)
        return loss/len(y)

    def fit(self, X, y, rate=1e-5):
        if X.shape[1] != self.feature_num:
            print "##"
            return
        # add a column _X : N * (feature_num + 1)
        _X = np.matrix([[1.]*(self.feature_num+1)] * len(X))
        _X[:,:-1] = X

        # initialize boolean matrix
        '''
        [ [ I{y_1==class_1}, I{y_2==class_1}, ..., I{y_n==class_1} ],
          [ I{y_1==class_2}, I{y_2==class_2}, ..., I{y_n==class_2} ],
          ...
          [ I{y_1==class_c}, I{y_2==class_c}, ..., I{y_n==class_c} ]
        ]
        '''
        boolean_mat = []
        for k in range(self.c):
            lst = []
            for i in range(len(y)):
                if int(y[i]) == k:
                    lst.append(1.0)
                else:
                    lst.append(0.0)
            boolean_mat.append(lst)
        boolean_mat = np.matrix(boolean_mat)

        # GD
        iter_num = 0
        loss_old, loss_new = 0, 10
        while iter_num < self.max_iter and abs(loss_old-loss_new)>1e-5:
            loss_old = loss_new
            decison_matrix = self.decison_function(_X)
            self.theta = self.theta + (boolean_mat - decison_matrix) * _X * rate
            # print self.theta
            loss_new = self.loss_function(decison_matrix, y)
            print "loop: " + str(iter_num), "loss: ", -loss_new
            iter_num += 1
        f_theta = open('theta.pkl', 'w')
        pickle.dump(self.theta, f_theta)
        f_theta.close()

    def predict(self, t_X):
        _t_X = np.matrix([[1.]*(self.feature_num+1)] * len(t_X))
        _t_X[:,:-1] = t_X
        pred_mat = self.decison_function(_t_X)
        pred_lst = pred_mat.argmax(axis=0).tolist()[0]
        return pred_lst

if __name__ == '__main__':
    sfx = SoftmaxClassifier(784, c=10, max_iter=150)
    import gzip
    f = gzip.open("mnist.pkl.gz","rb")
    train, dev, test = pickle.load(f)

    train_X = np.matrix(train[0])
    train_y = train[1].tolist()
    test_X = np.matrix(test[0])
    test_y = test[1].tolist()
    print test_X.shape
    print len(test_y)

    import time
    tic = time.time()

    rate=2*1e-5
    sfx.fit(train_X, train_y, rate=rate)

    preds = sfx.predict(test_X)

    assert len(preds) == len(test_y)
    accuracy = sum([int(a==b) for a,b in zip(preds, test_y)]) *1.0 / len(test_y)
    print accuracy
    print "rate:", rate
    print "iter_num:", sfx.max_iter

    print (time.time() - tic)/60.0, "mins"

