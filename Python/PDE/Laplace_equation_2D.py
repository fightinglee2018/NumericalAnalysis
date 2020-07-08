# encoding: utf-8

"""
Solving the 2-D Laplace's equation by the Finite Difference Method 
Numerical scheme used is a second order central difference in space (5-point difference)
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 60                               # Number of steps in space(x)
ny = 60                               # Number of steps in space(y)
niter = 10000                         # Number of iterations 
dx = 2.0 / (nx - 1)                   # Width of space step(x)
dy = 2.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 2, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 2, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

# Boundary Conditions
p[:, 0] = 0.0                         # Dirichlet condition
p[:, -1] = y                        # Dirichlet condition
p[0, :] = p[1, :]                     # Neumann condition
p[-1, :] = p[-2, :]               # Neumann condition

# Explicit iterative scheme with C.D in space (5-point difference)
# j = np.arange(1, nx-1)
# i = np.arange(1, ny-1)

for it in range(niter):
    pn = p.copy()
    p[1:-1, 1:-1] = ((pn[1:-1, 2:] + pn[1:-1, 0:-2])*dx*dx + (pn[2:, 1:-1] + pn[0:-2, 1:-1])*dy*dy) / (2.0 * (dx*dx + dy*dy))
    # p[1:-1, 1:-1] = (pn[1:-1, 2:] + pn[1:-1, 0:-2] + pn[2:, 1:-1] + pn[0:-2, 1:-1]) / 4   # Jacobi iter
    # Boundary condition
    p[:, 0] = 0.0                      # Dirichlet condition
    p[:, -1] = y                     # Dirichlet condition
    p[0, :] = p[1, :]                  # Neumann condition
    p[-1, :] = p[-2, :]            # Neumann condition

# print(p)

# Plot the solution
fig = plt.figure()      # Define new 3D coordinate system
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, p, cmap='rainbow')

plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
ax.set_xlabel("Spatial co-ordinate (x) ")
ax.set_ylabel(" Spatial co-ordinate (y)")
ax.set_zlabel("Solution profile (P) ")
plt.show()

