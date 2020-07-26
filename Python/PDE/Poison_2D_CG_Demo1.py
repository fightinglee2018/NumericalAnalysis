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


import time
import numpy as np
# import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# global parameters
nx = 100                               # Number of steps in space(x)
ny = 100                               # Number of steps in space(y)
max_iter = 1e5
niter = max_iter                         # Number of iterations 
dx = 1.0 / (nx - 1)                   # Width of space step(x)
dy = 1.0 / (ny - 1)                   # Width of space step(x)
x = np.linspace(0, 1, nx)             # Range of x(0,2) and specifying the grid points
y = np.linspace(0, 1, ny)             # Range of x(0,2) and specifying the grid points
# print(dx)

mu = 3.0
F = np.zeros((ny, nx))
for i in range(ny):
        F[i, :] = 5 * np.sin(x + y[i])
        # F[i, j] = 20 * np.cos(3*np.pi*x[j]) * np.sin(2*np.pi*y[i])

# F[0, 0] = 2*F[0, 0]
# F[0,-1] = 2*F[0, -1]
# F[-1, 0] = 2*F[-1, 0]
# F[-1, -1] = 2*F[-1, -1]

# f = np.reshape(F, (nx*ny,))

# f1 = np.sin(x)
# f2 = np.sin(y)
# f3 = np.sin(x+1)
# f4 = np.sin(y+1)

# g1 = -np.cos(x)
# g2 = -np.cos(y)
# g3 = np.cos(x+1)
# g4 = np.cos(y+1)

G = np.zeros((ny, nx))
# G[0, 1:-1] = -np.cos(x[1:-1])
# G[1:-1, 0] = -np.cos(y[1:-1])
# G[-1, 1:-1] = np.cos(x[1:-1]+1)
# G[1:-1, -1] = np.cos(y[1:-1]+1)
# G[0, 0] = G[0, 1] + G[1, 0]
# G[0, -1] = G[1, -1] + G[0, -2]
# G[-1, 0] = G[-1, 1] + G[-2, 0]
# G[-1, -1] = G[-1, -2] + G[-2, -1]

G[0, :] = -np.cos(x)
G[:, 0] = -np.cos(y)
G[-1, :] = np.cos(x+1)
G[:, -1] = np.cos(y+1)
G[0, 0] = 2*G[0, 0]
G[0, -1] = 2*G[0, -1]
G[-1, 0] = 2*G[-1, 0]
G[-1, -1] = 2*G[-1, -1]

# print(G)

# exact solution
U = np.zeros((ny, nx))
for i in range(ny):
    U[i, :] = np.sin(x + y[i])


# 
ttt = np.zeros((ny, nx))


# Inital Conditions
Un = np.zeros((ny, nx))
# Pn = np.zeros((ny, nx))

un = np.reshape(np.transpose(Un), (nx*ny,))

# Boundary Conditions
# p[:, 0] = f2                         # Dirichlet condition
# p[:, -1] = f4                        # Dirichlet condition
# p[0, :] = f1                     # Dirichlet condition
# p[-1, :] = f3               # Dirichlet condition

# p[0, 0] = (f[0, 0] * dx * dx + p[0, 1] + p[1, 0] - g1[0] * dy - g2[0] * dx) * 2 / (4 + mu*dx * dx)
# p[0, -1] = (f[0, -1] * dx * dx + p[1, -1] + p[0, -2] - g1[-1] * dy + g4[0] * dx) * 2 / (4 + mu*dx * dx)
# p[-1, 0] = (f[-1, 0] * dx * dx + p[-1, 1] + p[-2, 0] - g2[-1] * dx + g3[0] * dy) * 2 / (4 + mu*dx * dx)
# p[-1, -1] = (f[-1, -1] * dx * dx + p[-1, -2] + p[-2, -1] + g3[-1] * dy + g4[-1] * dx) * 2 / (4 + mu*dx * dx)
# p[1:-1, 0] = (f[1:-1, 0] * dx * dx + 2 * p[1:-1, 1] + p[:-2, 0] + p[2:, 0] - 2 * g2[1:-1] * dx) / (4 + mu*dx * dx)                   # Neumann condition
# p[1:-1, -1] = (f[1:-1, -1] * dx * dx + 2 * p[1:-1, -2] + p[:-2, -1] + p[2:, -1] + 2 * g4[1:-1] * dx) / (4 + mu*dx * dx)               # Neumann condition
# p[0, 1:-1] = (f[0, 1:-1] * dy * dy + 2 * p[1, 1:-1] + p[0, :-2] + p[0, 2:] - 2 * g1[1:-1] * dy) / (4 + mu*dx * dx)                   # Neumann condition
# p[-1, 1:-1] = (f[-1, 1:-1] * dy * dy + 2 * p[-2, 1:-1] + p[-1, :-2] + p[-1, 2:] + 2 * g3[1:-1] * dy) / (4 + mu*dx * dx)               # Neumann condition


def initial():
    r"""
    Initialize the x.
    """
    x = np.zeros(ny*nx)
    return x


def coefficient_mat():
    r"""
    Construct the coefficient matrix and b.
    """
    A = np.zeros((ny*nx, nx*ny), dtype=np.float)
    b = np.zeros(ny*nx, dtype=np.float)
    
    ## construct A
    # Build B
    # B = np.zeros((ny, nx))
    r = 0.5*(4 + mu*dx*dx) * np.ones(nx)
    r1 = -np.ones(nx-1)
    B = np.diag(r, 0) + np.diag(r1, 1) + np.diag(r1, -1)
    B[0, 1] = -2
    B[-1, -2] = -2
    # Build I
    I = np.eye(nx)
    # Build A
    A = np.kron(B, I) + np.kron(I, B)
    # Fix some values on boundary
    # A[0, 0] = A[0, 0] / 2
    # A[0, 1] = -1
    # A[0, ny] = -1
    # A[ny-1, ny-1] = A[ny-1, ny-1] / 2
    # A[ny-1, ny-2] = -1
    # A[ny-1, 2*ny-1] = -1
    # A[-ny, -ny] = A[-ny, -ny] / 2
    # A[-ny, -(ny-1)] = -1
    # A[-ny, -2*ny] = -1
    # A[-1, -1] = A[-1, -1] / 2
    # A[-1, -2] = -1
    # A[-1, -(ny+1)] = -1
    # idx = np.arange(ny, (nx-1)*ny, ny)
    # A[idx, idx-1] = 0
    # A[idx, idx+1] = -2
    # idx = np.arange(2*ny-1, nx*ny-1, ny)
    # A[idx, idx+1] = 0
    # A[idx, idx-1] = -2

    # print(A)

    ## Construct b(b = h^2 * f + 2 * h * g)
    # Build f
    f = np.reshape(np.transpose(F), nx*ny)
    # Build g
    g = np.reshape(np.transpose(G), nx*ny)
    # Build b
    b = dx*dx*f + 2*dx*g
    # b = dx*dx*f + g*dx
    
    return A, b


def cg(A, b, x):
    r"""
    Conjugate gradient method.
    Input:
        A: coefficient matrix
        b: right hand side
        x: the initial x
    Output:
        x: result of the linear equations Ax = b
    """
    r = b - np.dot(A, x)        # r = b-Ax
    p = np.copy(r)
    it = 0
    while np.max(np.abs(r)) > 1e-3 and it < max_iter:
        q = np.dot(A, p)
        pap = np.inner(p, q)
        if pap == 0:
            print("pap")
            print("iter: {}".format(it))
            print("e={}".format(np.max(np.abs(r))))
            return x
        alpha = np.inner(r, r) / pap
        r1 = r - alpha * q
        beta = np.inner(r1, r1) / np.inner(r, r)
        x1 = x + alpha * p
        p1 = r1 + beta * p
        r = r1
        p = p1
        x = x1
        it += 1

    print("iter:{}".format(it))
    # print("e={}".format(np.max(np.abs(r))))
    print("U_(n+1) - U_n: {}".format(np.max(np.abs(r))))

    return x


def compute_error(Un):
    # compute error
    # error = np.max(np.abs(Un - U))
    error = np.sqrt(np.sum(np.square(Un - U)) / (nx*ny))
    print("Error: {}".format(error))


def plot(Un, ttt):
    global x, y
    # Plot the solution
    fig = plt.figure()      # Define new 3D coordinate system
    ax = plt.axes(projection='3d')

    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, U, cmap='rainbow')            # plot u
    ax.plot_surface(x, y, Un, cmap='rainbow')            # plot p
    # ax.plot_surface(x, y, ttt, cmap='rainbow')            # plot ttt
    # ax.plot_surface(x, y, Un-U, cmap='rainbow')            # plot error

    plt.title("2-D Laplace equation; Number of iterations {}".format(niter))
    ax.set_xlabel("Spatial co-ordinate (x) ")
    ax.set_ylabel(" Spatial co-ordinate (y)")
    ax.set_zlabel("Solution profile (P) ")
    plt.show()


def main():
    start = time.time()

    A, b = coefficient_mat()    # Get A, b
    xx = initial()               # initial x
    ans = cg(A, b, xx)           # solve this linear system

    Un = np.transpose(np.reshape(ans, (ny, nx)))

    end = time.time()
    print("Cost time: {}".format(end - start))

    # Direct method
    # tt = np.dot(np.linalg.inv(A), b)
    # ttt = np.reshape(np.transpose(tt), (ny, nx))
    # # print(tt)

    # # print(ttt)
    # # print(Un)

    # eee = np.sum(np.abs(Un - ttt)) / (ny*nx)
    # print("eee = {}".format(eee))

    # compute_error(ttt)
    compute_error(Un)             # compute error

    plot(Un, ttt)                      # plot the solution

    # print("x = {}".format(ans))


if __name__ == '__main__':
    main()

