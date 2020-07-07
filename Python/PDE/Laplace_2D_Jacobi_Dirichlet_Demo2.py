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
nx = 60                               # Number of steps in space(x)
ny = 60                               # Number of steps in space(y)
niter = 10000                         # Number of iterations 
dx = 1.0 / (nx - 1)                   # Width of space step(x)
dy = 1.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 1, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 1, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

mu = 0
g = 1

f1 = x**2
f2 = -y**2
f3 = x**2 - 1
f4 = 1 - y**2

# exact solution
u = np.zeros((ny, nx))
for i in range(ny):
    u[i, :] = x**2 - y[i]**2

# for i in range(ny):
#     for j in range(nx):
#         u[i, j] = x[j]*x[j] - y[i]*y[i]

# Inital Conditions
p = np.zeros((ny, nx))
pn = np.zeros((ny, nx))

# Boundary Conditions
p[:, 0] = f2                         # Dirichlet condition
p[:, nx-1] = f4                        # Dirichlet condition
p[0, :] = f1                     # Dirichlet condition
p[ny-1, :] = f3               # Dirichlet condition

# Explicit iterative scheme with C.D in space (5-point difference)
# j = np.arange(1, nx-1)
# i = np.arange(1, ny-1)

e = 0

for it in range(niter):
    pn = p.copy()
    # p[1:ny-1, 1:nx-1] = ((pn[1:ny-1, 2:nx] + pn[1:ny-1, 0:nx-2])*dx*dx + (pn[2:ny, 1:nx-1] + pn[0:ny-2, 1:nx-1])*dy*dy) / (2.0 * (dx*dx + dy*dy))
    p[1:ny-1, 1:nx-1] = (pn[1:ny-1, 2:nx] + pn[1:ny-1, 0:nx-2] + pn[2:ny, 1:nx-1] + pn[0:ny-2, 1:nx-1]) / (4 + mu * dx * dx) # dx = dy
    # Boundary condition
    p[:, 0] = f2                         # Dirichlet condition
    p[:, nx-1] = f4                        # Dirichlet condition
    p[0, :] = f1                     # Dirichlet condition
    p[ny-1, :] = f3               # Dirichlet condition

    e = np.abs(p - pn).max()
    # if(e < 1e-5):
    #     break

print(e)
# print(p)

# compute error
# error = np.sum(np.abs(p - u)) / (ny*ny)
error = np.max(np.abs(p - u))
print(error)

# Plot the solution
fig = plt.figure()      # Define new 3D coordinate system
ax = plt.axes(projection='3d')

x, y = np.meshgrid(x, y)
ax.plot_surface(x, y, u, cmap='rainbow')          # plot u
ax.plot_surface(x, y, p, cmap='rainbow')          # plot p

plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
ax.set_xlabel("Spatial co-ordinate (x) ")
ax.set_ylabel(" Spatial co-ordinate (y)")
ax.set_zlabel("Solution profile (P) ")
plt.show()

