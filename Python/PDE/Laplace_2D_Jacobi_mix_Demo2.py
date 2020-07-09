# encoding: utf-8

r"""
Solving the 2-D Laplace's equation by the Finite Difference Method 
Numerical scheme used is a second order central difference in space (5-point difference)

PDE:
    u_xx + u_yy = 0         0<x<1, 0<y<1
    u_x(0, y) = 0, u(1, y) = 1-y^2    0<=y<=1
    u_y(x, 0) = 0, u(x, 1) = x^2-1    0<=x<=1
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 64                               # Number of steps in space(x)
ny = 64                               # Number of steps in space(y)
niter = 10000                         # Number of iterations 
dx = 1.0 / (nx - 1)                   # Width of space step(x)
dy = 1.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 1, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 1, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

g3 = -2
g4 = 2

# exact solution
u = np.zeros((ny, nx))
for i in range(ny):
    u[i, :] = x*x - y[i]*y[i]

# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

# Boundary Conditions
# p[-1, :] = x*x - 1                         # Dirichlet condition
# p[:, -1] = 1 - y*y                        # Dirichlet condition
p[:, 0] = p[:, 1]                     # Neumann condition
p[0, :] = p[1, :]               # Neumann condition

# p[:, 0] = - y*y                     # Dirichlet condition
# p[0, :] = x*x                       # Dirichlet condition
p[:, -1] = p[:, -2] + g4 * dx               # Neumann condition
p[-1, :] = p[-2, :] + g3 * dy               # Neumann condition

# Explicit iterative scheme with C.D in space (5-point difference)
e = 0
for it in range(niter):
    pn = p.copy()
    # p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dx*dx + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dy*dy) / (2.0 * (dx*dx + dy*dy))
    p[1:-1, 1:-1] = (pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]) / 4 # dx = dy
    # Boundary condition
    # p[-1, :] = x*x - 1                         # Dirichlet condition
    # p[:, -1] = 1 - y*y                        # Dirichlet condition
    p[:, 0] = p[:, 1]                     # Neumann condition
    p[0, :] = p[1, :]               # Neumann condition

    # p[:, 0] = - y*y                     # Dirichlet condition
    # p[0, :] = x*x                       # Dirichlet condition
    p[:, -1] = p[:, -2] + g4 * dx               # Neumann condition
    p[-1, :] = p[-2, :] + g3 * dy               # Neumann condition

    # is convergence
    e = np.abs(p - pn).max()

# print(p)
print(e)

# compute error
error = np.max(np.abs(p - u))
print(error)

# Plot the solution
fig = plt.figure()      # Define new 3D coordinate system
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)
# ax.plot_surface(x, y, u, cmap='rainbow')        # plot u
# ax.plot_surface(x, y, p, cmap='rainbow')        # plot p
ax.plot_surface(x, y, p - u, cmap='rainbow')    # plot error

plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
ax.set_xlabel("Spatial co-ordinate (x) ")
ax.set_ylabel(" Spatial co-ordinate (y)")
ax.set_zlabel("Solution profile (P) ")
plt.show()

