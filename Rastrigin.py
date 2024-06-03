# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:13:37 2024

@author: Ben
"""

import numpy as np
from math import pi
from math import cos
from math import sin
from scipy.optimize import minimize


#Requires a set of points
#Modifies Nothing
#Returns the value of the Rastrigin fuction at a point along with a tuple of its gradient 
def Rastrigin_2D(x,return_grad = True):
    f = 10*2+x[0]**2-10*cos(2*pi*x[0])+x[1]**2-10*cos(2*pi*x[1])
    if return_grad == False:
        return f
    else:
        df_dx = 20*pi*sin(2*pi*x[0])+2*x[0]
        df_dy = 20*pi*sin(2*pi*x[1])+2*x[1]
        df = np.array([df_dx,df_dy])
        return(f,df)

#Requires a set of points
#Modifies Nothing
#Returns the hessian matrix for 2D Rastrigin
def Rastrigin_2D_hess(x):
    hess = [[40*pi*pi*cos(2*pi*x[0])+2,0],
            [0,40*pi*pi*cos(2*pi*x[1])+2]]
    return hess

#Asks the user for desired methods and desired min
#Sets the initial guess to bottom right corner of the possible values
method = int(input("1 for CG and 2 for Newton CG: "))
req_min = float(input("input your desired min: "))
x0 = []
x0.append(-4.52)
x0.append(-4.52)

#CG Method, gets the initial minimize and moves through the grid until the minimum is met then returns
if method == 1:
    #initial setup, first minimum and stores the x value to reset it
    result = minimize(Rastrigin_2D, x0,jac=True, method='CG',options={'disp':False})
    current_min = result.fun
    store = x0[0]
    
    #while the minimum is not met moves across the grid in a given step size
    while(current_min > req_min):
        x0[0] = store
        x0[1] = x0[1]+0.5
        while(x0[0] < 4.52):
            result = minimize(Rastrigin_2D, x0,jac=True, method='CG',options={'disp':False})
    #if the minimum is met breaks out and returns the values
            if(result.fun < current_min):
                current_min = result.fun
                pos = result.x
                break
            x0[0] = x0[0]+0.5
        
    print("Optimal parameters:", pos)
    print("Minimum value:", current_min)
    
#Newton CG method, code functions the same with the inclusion of the hess function in the call
if method == 2:
    result = minimize(Rastrigin_2D, x0,jac=True, hess=Rastrigin_2D_hess, method='Newton-CG',options={'disp':False})
    current_min = result.fun
    store = x0[0]
    while(current_min > req_min):
        x0[0] = store
        x0[1] = x0[1]+0.5
        while(x0[0] < 4.52):
            result = minimize(Rastrigin_2D, x0,jac=True, hess=Rastrigin_2D_hess, method='Newton-CG',options={'disp':False})
            if(result.fun < current_min):
                current_min = result.fun
                pos = result.x
                break
            x0[0] = x0[0]+0.5
        
    print("Optimal parameters:", pos)
    print("Minimum value:", current_min)