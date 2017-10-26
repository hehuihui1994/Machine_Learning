# -*- coding: utf-8 -*-

'''
acc 0.999
'''

from sklearn import metrics
import cPickle as pickle
import math
import numpy as np

def read_data(data_file):
    import gzip
    f = gzip.open(data_file, "rb")
    train, val, test = pickle.load(f)
    f.close()
    #向量
    train_x = train[0]
    #加偏置
    ones = np.ones(train_x.shape[0])
    train_x =np.c_[ones,train_x]
    #标签直接是  输出一个整数
    train_y = train[1]
    test_x = test[0]
    #加偏置
    ones = np.ones(test_x.shape[0])
    test_x =np.c_[ones,test_x]
    test_y = test[1]
    return train_x, train_y, test_x, test_y

def hypothesis(thetas, X):
    return 1.0/ (1 + np.exp(-1 * X.dot(thetas)))

def mle(X, Y, thetas):
    f = hypothesis(thetas, X)
    return Y.dot(np.log(f + 1e-10))+ (1 - Y).dot(np.log(1- f + 1e-10))

def get_gradient(X, Y, thetas ,i):
    error = hypothesis(thetas, X[i,:]) - Y[i]
    return X[i, :].T.dot(error)

def z_score_normalization(X):
    for i in range(1, int(X.shape[1])):
        std, mean = np.std(X[:,i], ddof = 1), np.mean(X[:,i])
        X[:,i] = (X[:,i]- mean)/(std + 1e-10)
    return X

def GD(X, Y, thetas, rate = 0.1, max_iter = 2000, end_condition=1e-4):
    X = z_score_normalization(X)
    sum_old, sum_new = 0, -mle(X, Y, thetas)
    iteration = 0 
    while iteration < max_iter and abs(sum_old-sum_new) > end_condition:
        grads = X.T.dot(hypothesis(thetas, X) - Y)
        thetas -= rate * grads
        iteration += 1
        sum_old = sum_new
        sum_new = -mle(X, Y, thetas)
        print("loop: %r  loss: %r"%(iteration,sum_new))
    return thetas      

#取mnist中的0,1
def get_samples(_x, _y):
    x=[]
    y=[]
    for i in range(len(_y)):
        if _y[i] == 0 or _y[i] == 1:
            y.append(_y[i])
            x.append(_x[i])
    return np.array(x),np.array(y)  

if __name__ == '__main__':
	#读取数据
    data_file = "mnist.pkl.gz"
    #读入的时候加偏置
    train_x1, train_y1, test_x1, test_y1 = read_data(data_file)
    #只取样本中的0,1
    #10610train,2115test
    train_x, train_y = get_samples(train_x1, train_y1)
    test_x, test_y = get_samples(test_x1, test_y1)
    # print "train",len(train_x)
    # print "test",len(test_x)
    thetas=[0 for i in range(len(train_x[0]))]
    thetas = GD(train_x, train_y, thetas, 0.0001)
    #预测
    predict = []
    #预测前先归一化
    test_x = z_score_normalization(test_x)
    for x_sample in test_x:
        p = hypothesis(thetas,x_sample)
        if p>= (1-p):
            predict.append(1)
        else:
            predict.append(0)
    #acc
    accuracy = metrics.accuracy_score(test_y, predict)
    print ("acc %r"%(accuracy))