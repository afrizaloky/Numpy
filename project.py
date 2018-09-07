__author__      = "Afrizaloky"
__copyright__   = "Copyright 2018, Indonesia"

import numpy as np
import numpy.linalg as linalg

a = [[1,0,0],[0,1,0],[0,0,1]]
b = [[20,10,30],[35,25,35],[50,10,20]]
b_ = [10,20,10]

## Multiply
c = np.matmul(a,b)
print(c)

## Subtract
c = np.subtract(a,b)
print(c)

## Add
c = np.add(a,b)
print(c)

## Multiply
c = np.divide(a,b)
print(c)

# Inverse
c = np.linalg.inv(b)
print(c)

# Solving Equation
c = np.linalg.solve(b, b_) ## b.c = b_
print(c)
d = np.allclose(np.dot(b, c), b_)
print(d)

# Transpose
c = np.transpose(b)
print(c)

##