# -*- encoding: UTF-8 -*-
'''Recursive Taylor Coefficients
    Author: Dr. Alex Pronschinske
    Version: 1.0.0
    
    Description of this module.
'''

# third-party modules
import scipy as np

#==============================================================================
def const(k):
    c = np.zeros(k+1)
    c[0] = 1.0
    return c
# END const

#==============================================================================
def div(U, V, approx='none'):
    '''Evaluate derivatives of f(u,v) = u/v
    U: ndarray, taylor coefficients
    V: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'vlin')'''
    k = len(U) - 1
    Y = np.zeros(k+1)
    
    Y[0] = U[0]/V[0]
    
    if approx == 'none' and k > 0:
        for i in range(1, k+1):
            s = 0.0
            for j in range(1, i+1):
                s += V[j]*Y[i-j]
            Y[i] = (U[i] - s)/V[0]
    elif approx == 'vlin' and k > 0:
        for i in range(1, k+1):
            Y[i] = ( U[i] - V[1]*Y[i-1] )/V[0]
    
    return Y
# END div

#==============================================================================
def exp(U, approx='none', *args):
    '''Evaluate derivatives of f(u) = exp(u)
    U: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'lin', 'quad')
    order: int, optional, if given will trigger a single order return otherwise
        all orders up to the highest given in U are returned'''
    # If no k is given calculate all derivatives possible
    if len(args) == 0:
        k = len(U) - 1
        Y = np.zeros(k+1)
        
        Y[0] = np.exp(U[0])
        
        if approx == 'none' and k > 0:
            for i in range(1,k+1):
                for j in range(0,i):
                    Y[i] += (1 - j/i) * U[i-j] * Y[j]
        elif approx == 'lin' and k > 0:
            for i in range(1,k+1):
                Y[i] = U[1] * Y[i-1] / i
        elif approx == 'quad' and k > 0:
            Y[1] = U[1] * Y[0]
            if k > 1:
                for i in range(2,k+1):
                    Y[i] = ( 2.0*U[2]*Y[i-2] + U[1]*Y[i-1] )/i
        
        return Y
    # If k is given calculate only that derivative
    else:
        k = int(args[0])
        y = np.exp(U[0])
        
        if approx == 'lin' and k > 0:
            for i in range(1,k+1):
                y *= U[1] / i
        
        return y
# END exp

#==============================================================================
def gauss(X, x0=0.0, w=1.0):
    '''Evaluate derivatives of f(u) = exp( -ln(16) * ((x-x0)/w)^2 )
    X: ndarray, taylor coefficients, should be of the form [x, 1, 0, 0, ...]
    x0: float, optional, =0.0, distribution center
    w: float, optional, =1.0, distribution full-width at half-max'''
    c = const(len(X)-1)
    U = (-np.log(16.0)/w**2) * sqr(X - x0*c, approx='lin')
    Y = exp(U, approx='quad')
    
    return Y
# END gauss

#==============================================================================
def inv(U, approx='none'):
    '''Evaluate derivatives of f(u) = 1/u = u^-1
    U: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'lin')'''
    Y = np.zeros(len(U))
    Y[0] = 1.0/U[0]
    
    if approx == 'none':
        for k in range(1, len(U)):
            s = 0.0
            for j in range(1, k+1):
                s += U[j] * Y[k-j]
            Y[k] = -Y[0] * s
    elif approx == 'lin':
        for k in range(1, len(U)):
            Y[k] = -Y[0] * Y[k-1]
    # END if
    
    return Y
# END inv

#==============================================================================
def ln(U, approx='none', *args):
    '''Evaluate derivatives of f(u) = ln(u)
    U: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'lin')
    order: int, optional, if given will trigger a single order return otherwise
        all orders up to the highest given in U are returned'''
    k = len(U) - 1
    Y = np.zeros(k+1)
    
    Y[0] = np.log(U[0])
    
    if approx == 'none' and k > 0:
        for i in range(1,k+1):
            s = 0.0
            for j in range(1,i):
                s += (1 - j/i) * U[j] * Y[i-j]
            Y[i] = (U[i]-s) / U[0]
    elif approx == 'lin' and k > 0:
        Y[1] = U[1]/U[0]
        if k > 1:
            for i in range(2,k+1):
                Y[i] = (1-i) * U[1] * Y[i-1] / (U[0] * i)
    
    return Y
# END ln

#==============================================================================
def lorentz(X, x0=0.0, w=1.0):
    '''Evaluate derivatives of f(u) = 1 / ( 1 + 4*(x-x0)^2/w^2 )
    X: ndarray, taylor coefficients, should be of the form [x, 1, 0, 0, ...]
    x0: float, optional, =0.0, distribution center
    w: float, optional, =1.0, distribution full-width at half-max'''
    k = len(X) - 1
    Y = inv( const(k) + 4.0*sqr(X - x0*const(k), approx='lin')/w**2 )
    
    return Y
# END lorentz

#==============================================================================
def power(U, a, approx='none'):
    '''Evaluate derivatives of f(u) = u^a
    U: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'lin')'''
    
    if a == -1:
        return inv(U, approx=approx)
    elif a == 2:
        return sqr(U, approx=approx)
    
    k = len(U) - 1
    Y = np.zeros(k+1)
    Y[0] = pow(U[0], a)
    
    if approx == 'none' and k > 0:
        for i in range(0, k):
            s = 0.0
            for j in range(0, i-1):
                s += (a - (a+1)*j/i) * Y[j] * U[i-j]
            Y[i] = s/U[0]
    elif approx == 'lin' and k > 0:
        for i in range(1, k+1):
            Y[i] = ((a+1-i)/i) * U[1] * Y[i-1] / U[0]
    
    return Y
# END power

#==============================================================================
def prod(U, V, approx='none', k=None):
    '''Evaluate derivatives of f(u,v) = u*v
    U: ndarray, taylor coefficients
    V: ndarray, taylor coefficients
    approx: str, optional, ='none' (, 'ulin', 'vlin', 'uvlin')
    order: int, optional, if given will trigger a single order return otherwise
        all orders up to the highest given in U and V are returned'''
    
    if approx == 'vlin':
        utemp = U
        vtemp = V
        U = vtemp
        V = utemp
        approx = 'ulin'
    
    # If no k is given calculate all derivatives possible
    if k==None:
        k = len(U) - 1
        Y = np.zeros(k+1)
        
        if approx == 'none':
            for i in range(0, k+1):
                for j in range(0, i+1):
                    Y[i] += U[j] * V[i-j]
        elif approx == 'ulin':
            Y[0] = U[0]*V[0]
            if k > 0:
                for i in range(1, k+1):
                    Y[i] = U[0]*V[i] + U[1]*V[i-1]
        elif approx == 'uvlin':
            Y[0] = U[0]*V[0]
            if k > 0:
                Y[1] = U[0]*V[1] + U[1]*V[0]
                if k > 1:
                    Y[2] = U[1]*V[1]
        
        return Y
    # If k is given calculate only that derivative
    else:
        y = 0.0
        
        if approx == 'none':
            for i in range(0, k +1):
                y += U[i] * V[k-i]
        elif approx == 'ulin':
            if k == 0:
                y = U[0]*V[0]
            else:
                y = U[0]*V[k] + U[1]*V[k-1]
        elif approx == 'uvlin':
            if k == 0:
                y = U[0]*V[0]
            elif k == 1:
                y = U[0]*V[1] + U[1]*V[0]
            elif k == 2:
                y = U[1]*V[1]
            else:
                y = 0.0
        
        return y
# END prod

#==============================================================================
def sin(X, approx='none'):
    '''Evaluate derivatives of f(u) = sin(u)
    X: ndarray, taylor coefficient array
    approx: str, optional, ='lin' (, 'none')'''
    # Y = sin(X)
    Y = np.zeros(len(X))
    Y[0] = np.sin(X[0])
    # Z = dY/dt = d/dt( sin(x) ) = cos(X) dX/dt
    Z = np.zeros(len(X))
    Z[0] = np.cos(X[0])
    
    if approx == 'none':
        for k in range(1, len(X)):
            # calculate Y, sin
            sy = 0.0
            sz = 0.0
            for i in range(k):
                sy += (i+1) * Z[k-1-i] * X[i+1]
                sz += (i+1) * Y[k-1-i] * X[i+1]
            Y[k] = sy / k
            Z[k] = -sz / k
    elif approx == 'lin' and k > 0:
        Y[1] = X[1]*np.cos(X[0])
        if k > 1:
            for i in range(2, k+1):
                Y[i] = -X[1]**2 * Y[i-2] / (i*(i-1))
    
    return Y
# END sin

#==============================================================================
def cos(U, approx='none'):
    '''Evaluate derivatives of f(u) = cos(u)
    U: ndarray, taylor coefficient array
    approx: str, optional, ='lin' (, 'none')'''
    # Y = cos(X)
    Y = np.zeros(len(X))
    Y[0] = np.cos(X[0])
    # Z = dY/dt = d/dt( cos(x) ) = -sin(X) dX/dt
    Z = np.zeros(len(X))
    Z[0] = -np.sin(X[0])
    
    Y[0] = np.cos(U[0])
    
    if approx == 'none' and k > 0:
            # calculate Y, sin
            sy = 0.0
            sz = 0.0
            for i in range(k):
                sy += (i+1) * Z[k-1-i] * X[i+1]
                sz += (i+1) * Y[k-1-i] * X[i+1]
            Y[k] = -sy / k
            Z[k] = sz / k
    elif approx == 'lin' and k > 0:
        Y[1] = -U[1]*np.sin(U[0])
        if k > 1:
            for i in range(2, k+1):
                Y[i] = -U[1]**2 * Y[i-2] / (i*(i-1))
    
    return Y
# END cos

#==============================================================================
def tan(U, approx='none'):
    '''Evaluate derivatives of f(u) = tan(u)
    U: ndarray, taylor coefficient array
    approx: str, optional, ='lin' (, 'none')'''
    Y = np.zeros(len(U))
    Y[0] = np.tan(U[0])
    
    Y[1] = U[1]*np.cos(U[0])
    if k > 1:
        for i in range(2, k+1):
            Y[i] = -U[1]**2 * Y[i-2] / (i*(i-1))
    
    return Y
# END sin

#==============================================================================
def sqr(U, approx='none', *args):
    '''Evaluate derivatives of f(u) = u^2
    U: ndarray, taylor coefficient array
    approx: str, optional, ='none' (, 'lin')
    order: int, optional, if given will trigger a single order return otherwise
        all orders up to the highest given in U are returned'''
    # If no k is given calculate all derivatives possible
    if len(args) == 0:
        k = len(U) - 1
        Y = np.zeros(k+1)
        
        if approx == 'none':
            for i in range(0,k+1):
                for j in range(0,i+1):
                    Y[i] += U[j] * U[i-j]
        elif approx == 'lin':
            Y[0] = U[0]**2
            if k > 0:
                Y[1] = 2.0*U[1]*U[0]
                if k > 1:
                    Y[2] = U[1]**2
        
        return Y
    # If k is given calculate only that derivative
    else:
        k = args[0]
        y = 0.0
        
        if approx == 'none':
            for i in range(0,k+1):
                y += U[i] * U[k-i]
        elif approx == 'lin':
            if k == 0:
                y = U[0]**2
            elif k == 1:
                y = 2*U[1]*U[0]
            elif k == 2:
                y = U[1]**2
            else:
                y = 0.0
        
        return y
# END sqr

#==============================================================================
def sqrt(U, approx='none', *args):
    '''Evaluate derivatives of f(u) = sqrt(u)
    U: ndarray, taylor coefficient array
    approx: str, optional, ='none' (, 'lin')
    order: int, optional, if given will trigger a single order return otherwise
        all orders up to the highest given in U are returned'''
    # If no k is given calculate all derivatives possible
    if len(args) == 0:
        k = len(U) - 1
        Y = np.zeros(k+1)
        
        Y[0] = np.sqrt(U[0])
        
        if approx == 'none' and k > 0:
            for i in range(1,k+1):
                s = 0.0
                for j in range(0,i):
                    s += (i-3.0*j) * Y[j] * U[i-j]
                Y[i] = s/(2.0*i*U[0])
        elif approx == 'lin' and k > 0:
            for i in range(1,k+1):
                Y[i] = (3.0-2.0*k) * U[1] * Y[i-1] / (2.0*k*U[0])
        
        return Y
    # If k is given calculate only that derivative
    else:
        k = args[0]
        y = np.sqrt(U[0])
        
        if approx == 'lin' and k > 0:
            for i in range(1,k+1):
                y *= (3.0-2.0*k) * U[1] / (2.0*k*U[0])
        
        return y
# END sqrt

#==============================================================================
def test():
    N = 200
    X = np.array([ np.linspace(-6.5, 6.5, N),
                   np.ones(N),
                   np.zeros(N),
                   np.zeros(N),
                   np.zeros(N),
                 ]).T
    Y = np.zeros(X.shape)
    
    for j in range(X.shape[0]):
        Y[j,:] = sin(X[j,:])
    # END for
    X = X.T
    Y = Y.T
    
    import matplotlib.pyplot as plt
    from math import factorial as fac
    #fig, ax = plt.subplots(X.shape[0], 1)
    #for k in range(X.shape[0]):
    #    ax[k].plot(X[0,:], Y[k,:]*fac(k))
    fig, ax = plt.subplots()
    for k in range(X.shape[0]):
        ax.plot(X[0,:], Y[k,:]*fac(k))
    ax.set_ylim(-2, 2)
    plt.show()
    plt.close()
# END test

#==============================================================================
if __name__ == '__main__': test()




