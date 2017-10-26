# -*- coding: utf-8 -*-

'''
Analytical method
'''
from numpy import *
import matplotlib.pylab as pl

#readin,x[],y[]
def ReadIn(filename,x,y):
    file_object=open(filename)
    for line in file_object:
        lineS=line.split()
        xtemp=[]
        xtemp.append(1)
        xtemp.append(int(lineS[0]))
        x.append(xtemp) 
        y.append(float(lineS[1]))
  
#Hypothesis : h(x) = theta[0]+theta[1]*x1
def H(theta,x1):
    res=theta[0]+theta[1]*x1
    return res
      
#cost function : J(theta)
def J(theta,x,y):
    res=0
    for i in range(len(y)):
        res+=(H(theta,x[i][1]) - y[i])*(H(theta,x[i][1]) - y[i])
    return res/2
    
#Analytical method get theta   
def Analytical(x,y):
    X=mat(x)
    Y=mat(y).T
   # thetaTemp=(((X.T)*X).I)*(X.T)*Y
    thetaTemp1=dot(X.T,X)
    thetaTemp2=dot(thetaTemp1.I,X.T)
    thetaTemp3=dot(thetaTemp2,Y)
    #to list
    thetaTemp4=thetaTemp3.T
    thetaTemp5=thetaTemp4.tolist()
    theta=thetaTemp5[0]
    return theta
 
def show(x,y,theta):
    xx=[]
    for xi in x:
        xx.append(xi[1])
    pl.plot(xx,y,'o')
    yy=[H(theta,i) for i in xx]
    pl.plot(xx,yy,'y')
    pl.title('Nanjing Housing Price Prediction')
    pl.xlabel('Year')
    pl.ylabel('Price')
    pl.show()
        
        
    
    
    
if __name__ == '__main__':
    #x = [[] * 2 for i in range(200)]
    x=[]
    y=[]
    ReadIn('in.txt',x,y)
    theta=Analytical(x,y)
    print("Y=%f + %f * X\n"%(theta[0],theta[1]))
    print("cost : J(theta) = %r\n"%(J(theta,x,y)))
    result = H(theta,2014)
    print("the Nanjing housing price in 2014 is %r\n"%(result))
    show(x,y,theta)
   
    
    
    
    
    
    
    