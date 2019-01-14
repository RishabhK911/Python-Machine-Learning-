# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 17:40:06 2019

@author: admin
"""

#using * we do not need to call numpy agin and again or write numpy as np 
from numpy import *
points=genfromtxt('data.csv',delimiter=',')


def compute_error_forpoints(b,m,points):
    totalerror=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1] 
        totalerror += (y - (m * x + b)) ** 2
    return totalerror/float(len(points))  
        
def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

    


#hyperparameters
learning_rate=0.0001
# y=mx+b  the equaition of a straight line
initial_b=0
initial_m=0
no_iterations=1000
[b,m]=gradient_descent(points, initial_b, initial_m,learning_rate,no_iterations)
print(b)
print(m)


    
x=[]
y=[]
for i in range(0,len(points)):
    x.append(points[i,0])
    y.append(points[i,1])
    
y_pred=[]
for i in range(0,len(points)):
    t=m* x[i] + b
    y_pred.append(float(t))    
    
import matplotlib.pyplot as plt
plt.scatter(x,y, color = 'red')
plt.plot(x,y_pred ,color = 'blue')