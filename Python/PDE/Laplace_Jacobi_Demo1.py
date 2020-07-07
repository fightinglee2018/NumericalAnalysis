# encoding: utf-8

r"""
Solving the 2-D Laplace's equation by the Finite Difference Method 
Numerical scheme used is a second order central difference in space (5-point difference)

PDE:
    u_xx + u_yy + \mu u = 0
    \partial u /  \partial \nu = g
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 100                               # Number of steps in space(x)
ny = 100                               # Number of steps in space(y)
niter = 10000                         # Number of iterations 
dx = 2.0 / (nx - 1)                   # Width of space step(x)
dy = 2.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 2, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 2, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

mu = -2
# g = 1
g1 = np.cos(x)
g2 = np.cos(y)
g3 = np.cos(x+2)
g4 = np.cos(y+2)

f1 = np.sin(x)
f2 = np.sin(y)
f3 = np.sin(x + 2)
f4 = np.sin(y + 2)

# exact solution
u = np.zeros((ny, nx))
for i in range(ny):
    u[i, :] = np.sin(x + y[i])
print(u.shape)

# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

# Boundary Conditions
# p[:, 0] = 0.0                         # Dirichlet condition
# p[:, nx-1] = y                        # Dirichlet condition
# p[0, :] = p[1, :]                     # Neumann condition
# p[ny-1, :] = p[ny-2, :]               # Neumann condition

# p[:, 0] = f2                         # Dirichlet condition
# p[:, nx-1] = f4                        # Dirichlet condition
# p[0, :] = f1                     # Dirichlet condition
# p[ny-1, :] = f3               # Dirichlet condition

p[:, 0] = p[:, 1] + g2 * dx                         # Neumann condition
# print(p[:, 0])
p[:, nx-1] = p[:, nx-2] + g4 * dx                        # Neumann condition
p[0, :] = p[1, :] + g1 * dy                     # Neumann condition
p[ny-1, :] = p[ny-2, :] + g3 * dy               # Neumann condition

# Explicit iterative scheme with C.D in space (5-point difference)
# j = np.arange(1, nx-1)
# i = np.arange(1, ny-1)

for it in range(niter):
    pn = p.copy()
    # p[1:ny-1, 1:nx-1] = ((pn[1:ny-1, 2:nx] + pn[1:ny-1, 0:nx-2])*dx*dx + (pn[2:ny, 1:nx-1] + pn[0:ny-2, 1:nx-1])*dy*dy) / (2.0 * (dx*dx + dy*dy))
    p[1:ny-1, 1:nx-1] = (pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]) / (4 + mu * dx * dx) # dx = dy
    # Boundary condition
    # p[:, 0] = 0.0                      # Dirichlet condition
    # p[:, nx-1] = y                     # Dirichlet condition
    # p[0, :] = p[1, :]                  # Neumann condition
    # p[ny-1, :] = p[ny-2, :]            # Neumann condition

    # p[:, 0] = p[:, 1] + g2 * dx                         # Dirichlet condition
    # p[:, nx-1] = p[:, nx-2] + g4 * dx                        # Dirichlet condition
    # p[0, :] = p[1, :] + g1 * dy                     # Dirichlet condition
    # p[ny-1, :] = p[ny-2, :] + g3 * dy               # Dirichlet condition

    p[:, 0] = p[:, 1] + g2 * dx                         # Neumann condition
    p[:, nx-1] = p[:, nx-2] + g4 * dx                        # Neumann condition
    p[0, :] = p[1, :] + g1 * dy                     # Neumann condition
    p[ny-1, :] = p[ny-2, :] + g3 * dy               # Neumann condition

# print(p)

# compute the error
# error = (np.abs(p - u)).max()
error = np.max(np.abs(p - u))
print(error)


# Plot the solution
fig = plt.figure()      # Define new 3D coordinate system
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)

ax.plot_surface(x, y, u, cmap='rainbow')        # plot u
ax.plot_surface(x, y, p, cmap='rainbow')        # plot p

plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
ax.set_xlabel("Spatial co-ordinate (x) ")
ax.set_ylabel(" Spatial co-ordinate (y)")
ax.set_zlabel("Solution profile (P) ")
plt.show()

# p
# fig = plt.figure()      # Define new 3D coordinate system
# ax = plt.axes(projection='3d')

# x, y = np.meshgrid(x, y)

# ax.plot_surface(x, y, u, cmap='rainbow')
# ax.plot_surface(x, y, p, cmap='rainbow')

# plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
# ax.set_xlabel("Spatial co-ordinate (x) ")
# ax.set_ylabel(" Spatial co-ordinate (y)")
# ax.set_zlabel("Solution profile (P) ")
# plt.show()
