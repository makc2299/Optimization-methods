# Module block 2. Numerical methods of unconditional multidimensional minimization
import numpy as np
from numpy.linalg import norm, inv, det

# Zero order methods ##########################################
# 1. Coordinate descent method

def Dichotomy(x, i, eps):
    delta = eps/10.
    x_left = x.copy()
    x_right = x.copy()
    a = -10.0
    b = 10.0
    while abs(b-a) > eps:
        x_left[i] = (a +  b - delta)/2.
        x_right[i] = (a + b + delta)/2.
        if f(x_left) < f(x_right):
            b = x_right[i]
        else:
            a = x_left[i]
    return (a+b)/2

def CoordinateDescent(f, x0, eps):
    n = len(x0)
    x1 = np.zeros(n, dtype = np.float)
    for i in range(0,n):
        x1[i] = Dichotomy(x0, i, eps)
    k = 1
    while norm(x1 - x0, 1) > eps and k < 5000:
        x0 = x1.copy()
        for i in range(0, n):
            x1[i] = Dichotomy(x0, i, eps)
        k += 1
    return [x1, f(x1), k]

# 2. Nelder-Mead method

def NelderMead(f, x1, x2, x3, eps):
    alpha = 1.0
    beta = 0.5
    gamma = 2.0
    
    lst = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
    
    xl = np.array(lst[0][1])
    xs = np.array(lst[1][1])
    xh = np.array(lst[2][1])

    x4 = (xl + xs) / 2

    sigma = np.sqrt(1./3 * ((f(x1) - f(x4))**2 + (f(x2) - f(x4))**2 + (f(x3) - f(x4))**2))
    k = 0

    while (sigma > eps) & (k <= 250):
        
        flag = True
        x5 = x4 + alpha * (x4 - xh)
        if f(x5) <= f(xl): 
            x6 = x4 + gamma*(x5 - x4) 
            if f(x6) < f(xl):
                xh = x6
            else:
                xh = x5
        elif f(xs) < f(x5) and f(x5) <= f(xh): 
            x7 = x4 + beta*(xh - x4) 
            xh = x7
        elif f(xl) < f(x5) and f(x5) <= f(xs): 
            xh = x5
        else: 
            x1 = xl + 0.5 * (x1 - xl)
            x2 = xl + 0.5 * (x2 - xl) 
            x3 = xl + 0.5 * (x3 - xl) 
            flag = False

        if flag == True:
            x1 = xl
            x2 = xs
            x3 = xh

        lst = sorted([[f(x1), x1], [f(x2), x2], [f(x3), x3]])
      
        xl = np.array(lst[0][1])
        xs = np.array(lst[1][1])
        xh = np.array(lst[2][1])

        x4 = (xl + xs) / 2

        sigma = np.sqrt(1./3 * ((f(x1) - f(x4))**2 + (f(x2) - f(x4))**2 + (f(x3) - f(x4))**2))
        k += 1

    return [xl, f(xl), k]
###############################################################

# First order methods #########################################
#1. The method of the fastest gradient descent

def Dichotomy1(x0, eps):
    delta = eps/10.
    a = -2.0
    b =  2.5
    while np.abs(b-a) > eps:
        alpha1 = (a + b - delta)/2.
        alpha2 = (a + b + delta)/2.
        f1 = f(x0 - alpha1*grad(x0))
        f2 = f(x0 - alpha2*grad(x0))
        if f1 < f2:
            b = alpha2
        else:
            a = alpha1
    return (a + b)/2.

def GradientDescent(f, grad, x0, eps):
    alpha = Dichotomy1(x0, eps)
    x1 = x0 - alpha*grad(x0) 
    k = 1
    while norm((x1-x0), 1) > eps and k < 5000:
        x0 = x1
        alpha = Dichotomy1(x0, eps)
        x1 = x0 - alpha*grad(x0) 
        k = k + 1
    return [x1, f(x1), k]

# 2. The method of conjugate gradients
def ConjugateGradients(f, grad, x0, eps):
    p = -grad(x0)
    alpha = Dichotomy1(x0, eps)
    x1 = x0 + alpha*p
    k = 1
    while norm(x1-x0, 1) > eps and k < 5000:
        b =  (norm(grad(x1), 1))**2/(norm(grad(x0), 1))**2 
        p = -grad(x1) + b*p  
        x0 = x1
        alpha = Dichotomy1(x0, eps)
        x1 = x0 + alpha*p
        k = k + 1

    return [x1, f(x1), k]
###############################################################

# Second order methods ########################################

# 1. Newton's method
def Newton(f, grad, hesse, x0, eps):
    k = 0
    gr = grad(x0)
    while norm(gr, 1) > eps and k < 50:
        hs = inv(hesse(x0))
        dt1 = hs[0][0]
        dt2 = det(hs)
        if dt1 > 0 and dt2 > 0:
            p = -np.dot(hs , gr) 
        else:
            p = -gr 
        x1 = x0 + p
        k = k + 1
        x0 = x1
        gr = grad(x0)

    return [x1, f(x1), k]

# 2. The McWard method
def Marquardt(f, grad, hesse, x0, eps):
    k = 0
    E = np.eye(2)
    my = 10**3
    gr = grad(x0)
    while norm(gr, 1) > eps and k < 50:
        hs = hesse(x0)
        dod = np.dot(inv(hs + np.dot(my, E)), gr)
        x1 = x0 - dod
        if f(x1) < f(x0):
            my = my/2
        else:
            my = 2*my
        x0 = x1
        k = k + 1
        gr = grad(x0)

    return [x1, f(x1), k]

# # Test functions

# def f(x):
#     return 4*(x[0]-5)**2 + (x[1]-6)**2

# def grad(x):
#     return np.array([8*(x[0]-5), 2*(x[1]-6)])

# def hesse(x):
#     return np.array([[8., 0.], [0., 2.]])

# print(Marquardt(f, grad, hesse, np.array([-2., 2.]), 1e-6))