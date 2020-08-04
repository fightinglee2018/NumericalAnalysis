# encoding: utf-8

r"""
Solving the 2-D Laplace's equation by the Finite Difference Method 
Numerical scheme used is a second order central difference in space (5-point difference)

PDE:
    -(u_xx + u_yy) - 2u = 0      0<x<1, 0<y<1
    u(x, 0) = sin(x), u(x, 1) = sin(x+1)    0<=x<=1
    u(0, y) = sin(y), u(1, y) = sin(y+1)    0<=y<=1
exact solution:
    u(x, y) = sin(x+y)
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 100                               # Number of steps in space(x)
ny = 100                               # Number of steps in space(y)
niter = 100000                         # Number of iterations 
dx = 1.0 / (nx - 1)                   # Width of space step(x)
dy = 1.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 1, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 1, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

mu = -2.0
f = 0.0
f1 = np.sin(x)
f2 = np.sin(y)
f3 = np.sin(x+1)
f4 = np.sin(y+1)
f0 = 1

# exact fun
u = np.zeros((ny, nx))
for i in range(ny):
    u[i, :] = np.sin(x + y[i])

# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

# Boundary Conditions
p[:, 0] = f2                         # Dirichlet condition
p[:, -1] = f4                        # Dirichlet condition
p[0, :] = f1                     # Dirichlet condition
p[-1, :] = f3               # Dirichlet condition

# Explicit iterative scheme with C.D in space (5-point difference)
e = 0
for it in range(niter):
    pn = p.copy()
    # p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dy*dy + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dx*dx) / (2.0 * (dx*dx + dy*dy) + mu*dx*dx*dy*dy)
    p[1:-1, 1:-1] = (pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]) / (4 + mu * dx * dx) # dx = dy
    # Boundary condition
    p[:, 0] = f2                         # Dirichlet condition
    p[:, -1] = f4                        # Dirichlet condition
    p[0, :] = f1                     # Dirichlet condition
    p[-1, :] = f3               # Dirichlet condition

    # is convergence
    e = np.abs(p - pn).max()
    if e < 1e-10:
        break

# print(p)
print(e)

# compute error
error = np.max(np.abs(p - u))
# error = np.sum(np.abs(p - u)) / (ny*nx)
print(error)

# Plot the solution
fig = plt.figure()      # Define new 3D coordinate system
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, u, cmap='rainbow')            # plot u
ax.plot_surface(x, y, p, cmap='rainbow')            # plot p
# ax.plot_surface(x, y, p-u, cmap='rainbow')            # plot error

plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
ax.set_xlabel("Spatial co-ordinate (x) ")
ax.set_ylabel(" Spatial co-ordinate (y)")
ax.set_zlabel("Solution profile (P) ")
plt.show()

