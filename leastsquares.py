# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:32:12 2024

@author: Ben
"""

import numpy as np
#from math import pi
from math import cos
from math import sin
import random
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm

#Requires: the coffients and x value of a quadratic fuction
#Modifies: Nothing
#Effects: Returns the y value of the fuction
def base_func_poly(a,x):
    a0 = a[0]
    a1 = a[1]
    a2 = a[2]
    f = a0+a1*x+a2*x**2
    return f
    
#Requires: the coffients of a quadratic and the number of rows desired
#Modifies: Nothing
#Effects: Creates the A and b Matrix needed for Least Squares, returns them
# as a tuple 
def func_starting_mtx(a,m):
    x = random.uniform(-10,10)
    A_rb = np.array([1,x,x**2])
    b_rb = np.array(base_func_poly(a, x))
    
    for i in range(1,m):
        x = random.uniform(-10,10)
        A_rn = np.array([1,x,x**2])
        A_rb = np.vstack((A_rb,A_rn))
        
        b_rn = np.array((base_func_poly(a, x)))
        b_rb = np.vstack((b_rb,b_rn))

    return(A_rb,b_rb)
        
#Requires: a tuple of the starting Matrix
#Modifes: nothing
#Effects: Inputs noise into the starting matrix
def func_noise(t):
    A = t[0]
    b = t[1]
    
    noise = np.random.normal(0,1,len(b))
    b_noise = np.zeros((len(b),1))
    
    for i in range(0,len(b)):
        b_noise[i] = b[i]+noise[i]
        
    return (A,b_noise)

def least_square(t):
    A = t[0]
    b = t[1]
    A_t = np.transpose(A)
    At_b = np.matmul(A_t,b)
    At_A = np.matmul(A_t,A)
    At_A_inv = np.linalg.inv(At_A)
    beta = np.matmul(At_A_inv,At_b)
    return beta

def MSE_error(t,beta,non_lin):
    A_origin = t[0]
    b_origin = t[1]
    n = len(A_origin[0])
    MSE_sum = 0
    for i in range(0,n):
        if non_lin == True:
            x = A_origin[i,0]
        else:
            x= A_origin[i,1]
        if non_lin == True:
            y_pred = non_lin_func(beta, x)
        else:
            y_pred = base_func_poly(beta, x)
        MSE_sum = MSE_sum + (b_origin[i]-y_pred)**2
    MSE = MSE_sum/n
    return MSE

a= np.array([1,2,1])
m = 100
error_array = np.empty(m)


for i in range(0,m):
    start_mat = func_starting_mtx(a, m)

    use_mat = func_noise(start_mat)

    beta = least_square(use_mat)

    error = MSE_error(start_mat, beta,False)

    error_array[i] = error
    
error_array.sort()

std_error = statistics.stdev(error_array)

mean_error = statistics.mean(error_array)

#plt.plot(norm.pdf(error_array,mean_error,std_error))]

#Non Linear Section

def non_lin_func(a,x):
    return a[0]+a[1]*cos(a[2]*x+a[3])

def residual_m(a,x,y):
    return y-non_lin_func(a, x)

def derivative_a0(a,x):
    return 1.

def derivative_a1(a,x):
    x = x[0]
    return cos(a[2]*x+a[3])

def derivative_a2(a,x):
    x = x[0]
    f = -x*a[1]*sin(x*a[2]+a[3])
    f = float(f)
    return f

def derivative_a3(a,x):
    x = x[0]
    return float(-1*a[1]*sin(a[2]*x+a[3]))

def non_lin_xy_mat(a,m):
    x = random.uniform(-10,10)
    A_rb = np.array(x)
    R_rb = np.array(non_lin_func(a, x))
    
    for i in range(1,m):
        x = random.uniform(-10,10)
        A_rn = np.array(x)
        A_rb = np.vstack((A_rb,A_rn))
        
        R_rn = np.array(non_lin_func(a, x))
        R_rb = np.vstack((R_rb,R_rn))

    return(A_rb,R_rb) 

def non_lin_starting_mat(t,m,a):
    x = t[0]
    y = t[1]
    a0 = derivative_a0(a, x[0])
    a1= derivative_a1(a, x[0])
    a2= derivative_a2(a, x[0])
    a3= derivative_a3(a, x[0])
    
    A_rb = np.array([a0,a1,a2,a3])
    R_rb = np.array(y[0]-non_lin_func(a, x[0]))

    for i in range(1,m):
        A_rn = np.array([derivative_a0(a, x[i]), derivative_a1(a, x[i]), derivative_a2(a, x[i]), derivative_a3(a, x[i])])
        A_rb = np.vstack((A_rb,A_rn))
        
        R_rn = np.array(y[i]-non_lin_func(a, x[i]))
        R_rb = np.vstack((R_rb,R_rn))
    
    return(A_rb,R_rb)

m = 100
a_int = ([1,1,1,1])

xy = non_lin_xy_mat(a_int, m)

error_array2 = np.empty(m)

for i in range(1,m):
    xy_w_noise = func_noise(xy)

    delta_a = 1

    current_a = ([0.75,0.75,0.75,0.75])
    iterations = 0
    
    while (abs(delta_a) >= 0.05 or iterations  > 1000):
        AR = non_lin_starting_mat(xy_w_noise, m, current_a)
    
        beta = least_square(AR)
    
        a_hold = np.copy(current_a)
    
        current_a[0] = float(beta[0] +current_a[0])
        current_a[1] = float(beta[1] +current_a[1])
        current_a[2] = float(beta[2] +current_a[2])
        current_a[3] = float(beta[3] +current_a[3])
        iterations+= 1
    
        delta_a = sum(current_a) - sum(a_hold)
        
    error = MSE_error(xy, current_a,True)
    error_array2[i] = error
    
error_array2.sort()

std_error2= statistics.stdev(error_array2)

mean_error2 = statistics.mean(error_array2)

plt.plot(error_array2)