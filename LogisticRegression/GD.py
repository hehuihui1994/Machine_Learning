# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 16:35:28 2015

@author: huihui
"""

from numpy import *
import matplotlib.pylab as plt
import copy


#readin x[],y[]
def ReadIn(filename1,filename2,x,y):
    file_object1=open(filename1)
    for line in file_object1:
        lineS=line.split()
        xtemp=[]
        xtemp.append(1.0)
        xtemp.append(float(lineS[0]))
        xtemp.append(float(lineS[1]))
        x.append(xtemp)         
    file_object2=open(filename2)
    for line in file_object2:
        line=line.strip('\n')
        y.append(float(line))
  
#sigmoid
def sigmoid(x):
      return 1.0/(1+exp(-x))
      
# h（x）=g（theta[0]+theta[1]*x1+theta[2]*x2）  =1的概率
def H(theta,x):
    resTemp=theta[0]+theta[1]*x[1]+theta[2]*x[2]
    res=sigmoid(resTemp) 
    return res
    
#对数似然取负数   变成J（theta)  优化目标
def J(theta,x,y):
    sum=0.0
    for i in range(len(y)):
        sum+=(y[i]*log(H(theta,x[i]))+(1-y[i])*log(1-H(theta,x[i])))  
    return -sum
    
#GD
def Normalize(x):
    lenx=len(x)
    sum=0
    sum2=0
    max=0
    max2=0
    min=100000
    min2=100000
    for i in range(lenx):
        sum+=x[i][1]
        sum2+=x[i][2]
        if(x[i][1]>max):
            max=x[i][1]
        if(x[i][2]>max2):
            max2=x[i][2]
        
        if(x[i][1]<min):
            min=x[i][1]
        if(x[i][2]<min2):
            min2=x[i][2]
            
        
    mean1=(sum*1.0)/lenx
    mean2=(sum2*1.0)/lenx
    s1=max-min
    s2=max2-min2
    for i in range(lenx):
        x[i][1]=(x[i][1]-mean1)/s1
        x[i][2]=(x[i][2]-mean2)/s2
    return [x,mean2,s2]
    
def NewTheta(theta,x,y,a):
    tempSum0=0
    tempSum1=0
    tempSum2=0
    for i in range(len(y)):
        tempSum0+=H(theta,x[i])-y[i]
        tempSum1+=( H(theta,x[i])-y[i] )*x[i][1]
        tempSum2+=( H(theta,x[i])-y[i] )*x[i][2]
        
    theta[0]=theta[0]-a*tempSum0
    theta[1]=theta[1]-a*tempSum1
    theta[2]=theta[2]-a*tempSum2
    return theta



def GD(theta,x,y,a,mean2,s2,xOld):
    step=1
    JthetaOld=J(theta,x,y)
    print("loop: %r  loss: %r"%(step,JthetaOld))
    #show(xOld,x,y,theta,step,mean2,s2)
    step=step+1
    theta=NewTheta(theta,x,y,a)
    JthetaNew=J(theta,x,y)
    print("loop: %r  loss: %r"%(step,JthetaNew))
    show(xOld,x,y,theta,step,mean2,s2)
    step=step+1
    Jdlt=JthetaOld-JthetaNew   
    while(Jdlt>=0.001):
        JthetaOld=JthetaNew
        theta=NewTheta(theta,x,y,a)
        JthetaNew=J(theta,x,y)
        print("loop: %r  loss: %r"%(step,JthetaNew))
        show(xOld,x,y,theta,step,mean2,s2)
        #raw_input()
        step=step+1
        Jdlt=JthetaOld-JthetaNew
    return theta


#show 
def show(xOld,x,y,theta,step,mean2,s2):
    color=['b','y']
    x1=[xOld[i][1] for i in range(len(xOld))]
    x2=[xOld[i][2] for i in range(len(xOld))]
    cost=J(theta,x,y)
    #predict_x2
    predict_x2Temp=[-theta[0]/theta[2]-(theta[1]/theta[2])*x[i][1] for i in range(len(x))] 
    #均值归一化转换回去
    predict_x2=[ predict_x2Temp[i]*s2+mean2 for i in range(len( predict_x2Temp))]  
    #图1
    axarr[0].cla()
    for i in range(len(x)):  
        y[i]=int(y[i])
        axarr[0].scatter(x1[i],x2[i], color=color[y[i]])  
    axarr[0].plot(x1,predict_x2,'r')
    axarr[0].set_xlabel('Exam1')
    axarr[0].set_ylabel('Exam2')  
    #图2
    axarr[1].scatter(step,cost, color='r')       
    axarr[1].set_xlabel('iteration steps')
    axarr[1].set_ylabel('Cost value')
    axarr[1].set_xlim([0,150])
    axarr[1].set_ylim([0, 100])
    fig.canvas.draw()

        
    
if __name__ == '__main__':
    #读取数据
    x=[]
    y=[]
    ReadIn('examplex','exampley',x,y)
    #save x
    xOld=copy.deepcopy(x)
    #show  初始化两张空表
    fig, axarr = plt.subplots(1, 2, figsize=(10,5))
    fig.show()
    #for theta
    xTemp=Normalize(x)
    x=xTemp[0]
    mean2=xTemp[1]
    s2=xTemp[2]
    theta=[0,0,0]
    a=0.1
    theta=GD(theta,x,y,a,mean2,s2,xOld)
    raw_input()
   # show(xOld,x,y,theta,1,mean2,s2)
   