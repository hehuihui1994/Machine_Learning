# -*- coding: utf-8 -*-
"""
Created on Tue Dec 01 15:41:29 2015
Gradient
@author: huihui
"""

from numpy import *
import matplotlib.pylab as plt
import copy



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
    res=theta[0]+theta[1]*x1;
    return res
      
#cost function : J(theta)
def J(theta,x,y):
    res=0;
    for i in range(len(y)):
        res+=(H(theta,x[i][1]) - y[i])*(H(theta,x[i][1]) - y[i])
    return res/2
    
#gradient 
def Normalize(x):
    lenx=len(x)
    sum=0
    max=0
    min=100000
    for i in range(lenx):
        sum+=x[i][1]
        if(x[i][1]>max):
            max=x[i][1]
        if(x[i][1]<min):
            min=x[i][1]
        
    mean1=(sum*1.0)/lenx
    s1=max-min
    for i in range(lenx):
        x[i][1]=(x[i][1]-mean1)/s1
    xtemp=(2014-mean1)/s1
    return [x,xtemp]


def NewTheta(theta,x,y,a):
    tempSum0=0
    tempSum1=0
    for i in range(len(y)):
        tempSum0+=H(theta,x[i][1])-y[i]
        tempSum1+=( H(theta,x[i][1])-y[i] )*x[i][1]
        
    theta[0]=theta[0]-a*tempSum0
    theta[1]=theta[1]-a*tempSum1
    return theta

def gradient(xOld,theta,x,y,a):
    step=1
    show(xOld,x,y,theta,step)
    step=step+1
    JthetaOld=J(theta,x,y)
    print("Jtheta: %r"%(JthetaOld))
    #raw_input()
    theta=NewTheta(theta,x,y,a)
    show(xOld,x,y,theta,step)
    step=step+1
    JthetaNew=J(theta,x,y)
    print("Jtheta: %r"%(JthetaNew))
    #raw_input()
    Jdlt=JthetaOld-JthetaNew   
    while(Jdlt>=0.00000001):
        JthetaOld=JthetaNew
        theta=NewTheta(theta,x,y,a)
        show(xOld,x,y,theta,step)
        step=step+1
        JthetaNew=J(theta,x,y)
        print("Jtheta: %r"%(JthetaNew))
       # raw_input()
        Jdlt=JthetaOld-JthetaNew
        #print("Jdlt: %r"%(Jdlt))
       
    
    return theta
  

#show 
def show(xOld,x,y,theta,step):
    xx=[]
    for xi in xOld:
        xx.append(xi[1])
        
    yy=[H(theta,x[i][1]) for i in range(len(x))] 
    cost=J(theta,x,y)
    #图1
    axarr[0].cla()
    axarr[0].plot(xx, y, 'o', xx, yy, 'y')
    axarr[0].set_title('Nanjing Housing Price Prediction')
    axarr[0].set_xlabel('Year')
    axarr[0].set_ylabel('Price')  
    #图2
    axarr[1].scatter(step,cost, color='r')       
    axarr[1].set_xlabel('steps')
    axarr[1].set_ylabel('Cost value')
    axarr[1].set_xlim([0, 100])
    axarr[1].set_ylim([0, 150])
    axarr[1].set_title('Cost')
    fig.canvas.draw()

    
if __name__ == '__main__':
    #x = [[] * 2 for i in range(200)]
    x=[]
    y=[]
    ReadIn('in.txt',x,y)
    #save x
    xOld=copy.deepcopy(x)
    #show  初始化两张空表
    fig, axarr = plt.subplots(1, 2, figsize=(10,5))
    fig.show()
    #for theta
    theta=[3,3]
    a=0.1
    xtemp=Normalize(x)
    x=xtemp[0]
    xres=xtemp[1]
    theta=gradient(xOld,theta,x,y,a)
   #result
    print("Y=%f + %f * X\n"%(theta[0],theta[1]))
    print("cost : J(theta) = %r\n"%(J(theta,x,y)))
    result = H(theta,xres)
    print("the Nanjing housing price in 2014 is %r\n"%(result))
    raw_input()
 
    
    
    
    
    
    
    
    
   