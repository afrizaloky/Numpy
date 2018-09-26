from docutils.nodes import inline

__author__      = "Afrizaloky"
__copyright__   = "Copyright 2018, Indonesia"

import numpy as np
import numpy.linalg as linalg
from sympy import *

'''''
a = [[1,0,0],[0,1,0],[0,0,1]]
b = [[20,10,30],[35,25,35],[50,10,20]]
c = [10,20,10]
d = [3,5,6]

## Multiply
data_1 = np.matmul(a, b)
print(data_1)

## Subtract
data_1 = np.subtract(a, b)
print(data_1)

## Add
data_1 = np.add(a, b)
print(data_1)

## Multiply
data_1 = np.divide(a, b)
print(data_1)

# Inverse
data_1 = np.linalg.inv(b)
print(data_1)

# Solving Equation
data_1 = np.linalg.solve(b,c) ## b.c = b_
print(data_1)
d = np.allclose(np.dot(b, data_1), c)
print(d)

# Transpose
data_1 = np.transpose(b)
print(data_1)

## Determinan
data_1 = linalg.det(b)
print(data_1)

## Tokopedia 1

n = int(input("Enter a number: "))

for i in range(n):
    print ("0 " * (i) + " ".join(map(str, range(i + 1, n + 1))) )


## Pertidaksamaan Linear
import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve
from sympy import Symbol


def f1(x):
    return 4.0*x-2.0
def f2(x):
    return 0.5*x+2.0
def f3(x):
    return -0.3*x+7.0

x = Symbol('x')
x1, =  solve(f1(x)-f2(x))
x2, =  solve(f1(x)-f3(x))
x3, =  solve(f2(x)-f3(x))

y1 = f1(x1)
y2 = f1(x2)
y3 = f2(x3)

plt.plot(x1,f1(x1),'go',markersize=10)
plt.plot(x2,f1(x2),'go',markersize=10)
plt.plot(x3,f2(x3),'go',markersize=10)

plt.fill([x1,x2,x3,x1],[y1,y2,y3,y1],'red',alpha=0.5)

xr = np.linspace(0.5,7.5,100)
y1r = f1(xr)
y2r = f2(xr)
y3r = f3(xr)

plt.plot(xr,y1r,'k--')
plt.plot(xr,y2r,'k--')
plt.plot(xr,y3r,'k--')

plt.xlim(0.5,7)
plt.ylim(2,8)

plt.show()


## Pertidaksamaan
import numpy as np
import matplotlib.pyplot as plt


# Construct lines
# x > 0
x = np.linspace(0, 20, 2000)
# y >= 2
y1 = (x*0) + 2
# 2y <= 25 - x
y2 = (25-x)/2.0
# 4y >= 2x - 8
y3 = (2*x-8)/4.0
# y <= 2x - 5
y4 = 2 * x -5

# Make plot
plt.plot(x, y1, label=r'$y\geq2$')
plt.plot(x, y2, label=r'$2y\leq25-x$')
plt.plot(x, y3, label=r'$4y\geq 2x - 8$')
plt.plot(x, y4, label=r'$y\leq 2x-5$')
plt.xlim((0, 16))
plt.ylim((0, 11))
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')

# Fill feasible region
y5 = np.minimum(y2, y4)
y6 = np.maximum(y1, y3)
plt.fill_between(x, y5, y6, where=y5>y6, color='grey', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

## Symp expand
x = Symbol('x')
y = Symbol('y')
a = expand((x+y)**6)
print (a)
'''
## Symp Turunan
x = Symbol('x')
y = Symbol('y')
a = diff(6*x**5, x)
print (a)

