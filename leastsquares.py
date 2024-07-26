# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:32:12 2024

@author: Ben
"""

import numpy as np
from math import pi
from math import cos
from math import sin
from math import sqrt
import random
import matplotlib.pyplot as plt
import statistics
from scipy.optimize import minimize

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

def chi_square(a, t, return_grad=True):
    x = t[0]
    y = t[1]
    n = len(x)
    
    chi = 0
    
    # Computing chi-square
    for i in range(n):
        chi += (y[i] - non_lin_func(a, x[i]))**2
    
    # Computing gradients
    d_a0 = 0
    d_a1 = 0
    d_a2 = 0
    d_a3 = 0
    
    for i in range(n):
        term = cos(a[2]*x[i] + a[3])
        d_a0 += -2 * (-a[0] + y[i] + a[1] * term)
        d_a1 += 2 * term * (term * a[1] + y[i] - a[0])
        d_a2 += -2 * a[1] * x[i] * sin(x[i] * a[2] + a[3]) * (a[1] * term + y[i] - a[0])
        d_a3 += -2 * a[1] * sin(a[3] + a[2] * x[i]) * (a[1] * term + y[i] - a[0])
    
    d_chi = np.array([d_a0, d_a1, d_a2, d_a3])
    
    # Returning results based on return_grad flag
    if return_grad is False:
        return chi
    else:
        return chi, d_chi


def non_lin_xy_mat(a,m):
    x = random.uniform(-10,10)
    A_rb = np.array(x)
    R_rb = np.array(non_lin_func(a, x))
    
    for i in range(1,m):
        x = random.uniform(0,4*pi)
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

m = 10
a_int = ([1,1,1,1])

xy = non_lin_xy_mat(a_int, m)

error_array2 = np.empty(m)

a0_array = np.empty(m)
a1_array = np.empty(m)
a2_array = np.empty(m)
a3_array = np.empty(m)

for i in range(1,m):
    xy_w_noise = func_noise(xy)

    delta_a = 1

    current_a = [0.75,0.75,0.75,0.75]
    result = minimize(chi_square, current_a, jac=True, args=(xy_w_noise), method = 'CG')
    current_a = result.x
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
    
    a0_array[i] = current_a[0]
    a1_array[i] = current_a[1]
    a2_array[i] = current_a[2]
    a3_array[i] = current_a[3]
    error = MSE_error(xy, current_a,True)
    error_array2[i] = error
    
error_array2.sort()
a0_array.sort()
a1_array.sort()
a2_array.sort()
a3_array.sort()

a0_std = statistics.stdev(a0_array)
a1_std = statistics.stdev(a1_array)
a2_std = statistics.stdev(a2_array)
a3_std = statistics.stdev(a3_array)

a0_mean = statistics.mean(a0_array)
a1_mean = statistics.mean(a1_array)
a2_mean = statistics.mean(a2_array)
a3_mean = statistics.mean(a3_array)

a0_ci = (a0_mean-1.96*(a0_std/sqrt(m)),a0_mean+1.96*(a0_std/sqrt(m)))
a1_ci = (a1_mean-1.96*(a1_std/sqrt(m)),a1_mean+1.96*(a1_std/sqrt(m)))
a2_ci = (a2_mean-1.96*(a2_std/sqrt(m)),a2_mean+1.96*(a2_std/sqrt(m)))
a3_ci = (a3_mean-1.96*(a3_std/sqrt(m)),a3_mean+1.96*(a3_std/sqrt(m)))

std_error2= statistics.stdev(error_array2)

mean_error2 = statistics.mean(error_array2)