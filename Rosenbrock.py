# -*- coding: utf-8 -*-
"""
Created on Tue May 14 08:57:45 2024

@author: Ben
"""


import numpy as np
from scipy.optimize import minimize

def Rosenbrock_2D(x, return_grad = True):
    a = 1
    b = 100
    f = (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    if return_grad == False:
       return f
    else:
       df_dx0 = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
       df_dx1 = 2 * b * (x[1] - x[0]**2) 
       df = np.array([df_dx0, df_dx1])
       return (f,df)

def Rosenbrock_2D_hess(x):
    b = 100
    hess = [[2*(b+1),-2*b],[-2*b, 2*b]]
    return hess
    

def Rosenbrock_4D(x, return_grad = True):
    f = sum((100*x[i+1]-x[i]**2)**2+(1-x[i])**2 for i in range(3))
    if return_grad == False:
        return f
    else:
        grad = np.zeros_like(x)
        for i in range(3):
            grad[i] = -400 * (x[i + 1] - x[i]**2) * x[i] - 2 * (1 - x[i])
        for i in range(1, 4):
            grad[i] += 200 * (x[i] - x[i - 1]**2)
        return (f,grad)

def Rosenbrock_4D_hess(x):
    hess = [[1200*x[0]**2-400*x[1]+2,-400*x[0],0,0],
            [-400*x[0],1200*x[1]**2-400*x[2]+202,-400*x[1],0],
            [0,-400*x[1],1200*x[2]**2-400*x[3]+202,-400*x[2]],
            [0,0,-400*x[2],200]]
    return hess

deg = int(input("2d or 4d: "))
method = int(input("1 for CG and 2 for Newton CG"))
x0 = []
if(method == 1):
    if(deg == 2):
        for i  in range(0,deg):
            temp = int(input("guess: "))
            x0.append(temp)
        result = minimize(Rosenbrock_2D, x0,jac=True,method='CG',options={'disp':True})
        print("Optimal parameters:", result.x)
        print("Minimum value:", result.fun)
    if(deg == 4):
        for i  in range(0,deg):
            temp = int(input("guess: "))
            x0.append(temp)
        result = minimize(Rosenbrock_4D, x0,jac=True,method='CG',options={'disp':True})
        print("Optimal parameters:", result.x)
        print("Minimum value:", result.fun)
if(method == 2):
    if(deg == 2):
        for i  in range(0,deg):
            temp = int(input("guess: "))
            x0.append(temp)
        result = minimize(Rosenbrock_2D, x0,jac=True,hess=Rosenbrock_2D_hess ,method='Newton-CG',options={'disp':True})
        print("Optimal parameters:", result.x)
        print("Minimum value:", result.fun)
    if(deg == 4):
        for i  in range(0,deg):
            temp = int(input("guess: "))
            x0.append(temp)
        result = minimize(Rosenbrock_4D, x0,jac=True,hess=Rosenbrock_4D_hess ,method='Newton-CG',options={'disp':True})
        print("Optimal parameters:", result.x)
        print("Minimum value:", result.fun)