# encoding: utf-8

r"""
Solving the 2-D Laplace's equation by the Finite Difference Method 
Numerical scheme used is a second order central difference in space (5-point difference)

PDE:
    -(u_xx + u_yy) + \mu u = 3sin(x + y)             0<x<1, 0<y<1
    BCS:
        \partial u /  \partial \nu = g      0<=x<=1, 0<=y<=1
    or:
        u_y(x, 0) = sin(x), u_y(x, 1) = sin(x+1)    0<=x<=1
        u_x(0, y) = sin(y), u_x(1, y) = sin(y+1)    0<=y<=1

exact solution:
    u(x, y) = sin(x+y)      \mu = 1
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 20                               # Number of steps in space(x)
ny = 20                               # Number of steps in space(y)
niter = 10000                         # Number of iterations 
dx = 1.0 / (nx - 1)                   # Width of space step(x)
dy = 1.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 1, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 1, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

mu = 1.0
f = np.zeros((ny, nx))
for i in range(ny):
    for j in range(nx):
        f[i, j] = 3 * np.sin(x[j] + y[i])
f1 = np.sin(x)
f2 = np.sin(y)
f3 = np.sin(x+1)
f4 = np.sin(y+1)

g1 = np.cos(x)
g2 = np.cos(y)
g3 = np.cos(x+1)
g4 = np.cos(y+1)

# exact solution
u = np.zeros((ny, nx))
for i in range(ny):
    u[i, :] = np.sin(x + y[i])

F = np.reshape(f, (nx*ny,))


# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

P = np.reshape(p, (nx*ny,))

# Boundary Conditions
# p[:, 0] = f2                         # Dirichlet condition
# p[:, -1] = f4                        # Dirichlet condition
# p[0, :] = f1                     # Dirichlet condition
# p[-1, :] = f3               # Dirichlet condition

p[0, 0] = (f[0, 0] * dx * dx + p[0, 1] + p[1, 0] - g1[0] * dy - g2[0] * dx) * 2 / (4 + mu*dx * dx)
p[0, -1] = (f[0, -1] * dx * dx + p[1, -1] + p[0, -2] - g1[-1] * dy + g4[0] * dx) * 2 / (4 + mu*dx * dx)
p[-1, 0] = (f[-1, 0] * dx * dx + p[-1, 1] + p[-2, 0] - g2[-1] * dx + g3[0] * dy) * 2 / (4 + mu*dx * dx)
p[-1, -1] = (f[-1, -1] * dx * dx + p[-1, -2] + p[-2, -1] + g3[-1] * dy + g4[-1] * dx) * 2 / (4 + mu*dx * dx)
p[1:-1, 0] = (f[1:-1, 0] * dx * dx + 2 * p[1:-1, 1] + p[:-2, 0] + p[2:, 0] - 2 * g2[1:-1] * dx) / (4 + mu*dx * dx)                   # Neumann condition
p[1:-1, -1] = (f[1:-1, -1] * dx * dx + 2 * p[1:-1, -2] + p[:-2, -1] + p[2:, -1] + 2 * g4[1:-1] * dx) / (4 + mu*dx * dx)               # Neumann condition
p[0, 1:-1] = (f[0, 1:-1] * dy * dy + 2 * p[1, 1:-1] + p[0, :-2] + p[0, 2:] - 2 * g1[1:-1] * dy) / (4 + mu*dx * dx)                   # Neumann condition
p[-1, 1:-1] = (f[-1, 1:-1] * dy * dy + 2 * p[-2, 1:-1] + p[-1, :-2] + p[-1, 2:] + 2 * g3[1:-1] * dy) / (4 + mu*dx * dx)               # Neumann condition

# Explicit iterative scheme with C.D in space (5-point difference)
e = 0.0
for it in range(niter):
    pn = p.copy()
    # p[1:-1, 1:-1] = (f[1:-1, 1:-1] * dx*dx *dy*dy + (pn[1:-1, 2:] + pn[1:-1, 0:-2])*dy*dy + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dx*dx) / (2.0 * (dx*dx + dy*dy) + mu*dx*dx*dy*dy)
    # p[1:-1, 1:-1] = (f[1:-1, 1:-1] * dx*dx + pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]) / (4 + mu * dx * dx) # dx = dy
    p[1:-1, 1:-1] = (f[1:-1, 1:-1] * dx*dx + pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1] - mu*dx*dx*pn[1:-1, 1:-1]) / 4 # dx = dy
    # p[1:-1, 1:-1] = (f[1:-1, 1:-1] * dx*dx + p[1:-1, 2:] + p[1:-1, 0:-2] + p[2:, 1:-1] + p[0:-2, 1:-1] - mu*dx*dx*pn[1:-1, 1:-1]) / 4 # dx = dy
    # Boundary condition
    # p[:, 0] = f2                         # Dirichlet condition
    # p[:, -1] = f4                        # Dirichlet condition
    # p[0, :] = f1                     # Dirichlet condition
    # p[-1, :] = f3               # Dirichlet condition

    p[0, 0] = (f[0, 0] * dx * dx + p[0, 1] + p[1, 0] - g1[0] * dy - g2[0] * dx) * 2 / (4 + mu*dx * dx)
    p[0, -1] = (f[0, -1] * dx * dx + p[1, -1] + p[0, -2] - g1[-1] * dy + g4[0] * dx) * 2 / (4 + mu*dx * dx)
    p[-1, 0] = (f[-1, 0] * dx * dx + p[-1, 1] + p[-2, 0] - g2[-1] * dx + g3[0] * dy) * 2 / (4 + mu*dx * dx)
    p[-1, -1] = (f[-1, -1] * dx * dx + p[-1, -2] + p[-2, -1] + g3[-1] * dy + g4[-1] * dx) * 2 / (4 + mu*dx * dx)
    p[1:-1, 0] = (f[1:-1, 0] * dx * dx + 2 * p[1:-1, 1] + p[:-2, 0] + p[2:, 0] - 2 * g2[1:-1] * dx) / (4 + mu*dx * dx)                   # Neumann condition
    p[1:-1, -1] = (f[1:-1, -1] * dx * dx + 2 * p[1:-1, -2] + p[:-2, -1] + p[2:, -1] + 2 * g4[1:-1] * dx) / (4 + mu*dx * dx)               # Neumann condition
    p[0, 1:-1] = (f[0, 1:-1] * dy * dy + 2 * p[1, 1:-1] + p[0, :-2] + p[0, 2:] - 2 * g1[1:-1] * dy) / (4 + mu*dx * dx)                   # Neumann condition
    p[-1, 1:-1] = (f[-1, 1:-1] * dy * dy + 2 * p[-2, 1:-1] + p[-1, :-2] + p[-1, 2:] + 2 * g3[1:-1] * dy) / (4 + mu*dx * dx)               # Neumann condition


    # is convergence
    e = np.abs(p - pn).max()
    # if e < 1e-10:
    #     break


# print(p)
print(e)

# compute error
# error = np.max(np.abs(p - u))
error = np.sqrt(np.sum(np.square(p - u)) / (nx*ny))
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

