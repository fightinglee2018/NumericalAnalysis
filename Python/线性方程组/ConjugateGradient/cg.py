# encoding: utf-8

r"""
Soving the linear equations Ax = b by conjugate gradient method.
"""


import time
import numpy as np


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
    while np.max(np.abs(r)) > 1e-10 and it < 1e2:
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
    print("e={}".format(np.max(np.abs(r))))

    return x


def main():
    n = 10
    A = np.array([
        [1., 1., 0.,  0.,  0.,  0,  0.,  0.,  0.,  0], 
        [1., 1., 1. , 0.,  0,   0., 0.,  0.,  0.,  0], 
        [ 0., 1.,  1. , 1., 0.,  0., 0.,  0.,  0.,  0], 
        [ 0.,  0., 1., 1., 1., 0., 0.,  0. , 0.,  0], 
        [ 0.,  0.,  0., 1.,  1., 1.,0.,   0.,  0.,  0], 
        [ 0.,  0.,  0.,  0.,  1., 1.,1., 0.,  0.,  0], 
        [ 0.,  0. , 0. , 0. , 0., 1., 1., 1., 0., 0], 
        [ 0.,  0.,  0. , 0. , 0., 0.,  1., 1.,  1.,0], 
        [ 0.,  0. , 0.,  0,  0.,  0.,  0.,  1.,1., 1.], 
        [ 0.,  0. , 0. , 0, 0.,   0.,  0.,  0.,  1.,1], 
    ])
    b = np.array([2., 3., 3.,3., 3.,3., 3., 3., 3., 2.])
    x = np.zeros((n,), dtype=np.float)

    ans = cg(A, b, x)
    # print("r:{}".format(r))
    print("x={}".format(ans))


if __name__ == '__main__':
    main()