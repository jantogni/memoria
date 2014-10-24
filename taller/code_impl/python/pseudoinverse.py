from __future__ import division
import numpy as np
import math as ma
import traceback
import sys
import matplotlib.pyplot as plt
import random
import time

# Rows
rows = 3000

# Columns
cols = 2000

# Matrix Rank
rank = 1632

# Beta
beta = 1.0 / 9.0

# Size
size = 1


def createA(rows, cols, rank):
    A = np.random.rand(rows, cols)
    u, s, v = np.linalg.svd(A)

    S = np.zeros((rows, cols))

    new_sigma = np.random.uniform(0.001, 0.1, rank)
    new_sigma.sort()
    new_sigma = size * new_sigma[::-1]
    S[:rank, :rank] = np.diag(new_sigma)

    new_A = np.dot(u, np.dot(S, v))

    return new_A


def lambda_svd(A, rank):
    u, s, d = np.linalg.svd(A)

    lsvd = beta * s[rank - 1] * s[rank - 1]

    return lsvd


def sigma_svd(A, rank):
    u, s, d = np.linalg.svd(A)

    return s[rank - 1]


def concatMatrix(a, l):
    l_i = l * np.identity(cols)
    return np.concatenate((a, l_i))


def plot_vector(v, n):
    plt.plot(range(1, n + 1), v, 'ro')
    plt.show()


def to_array(v):
    vector = []
    for i in v:
        vector.append(i[0])
    return np.asarray(vector)


def iteration(A, b, l, n, mode):
    # Step 0, qr factorization of C, and create Q_1, -inverse(rows+lI)
    try:
        C = concatMatrix(A, ma.pow(l, 0.5))
        q, r = np.linalg.qr(C)
        q1 = q[0:rows, 0:cols]
        r1 = r[0:rows, 0:cols]

        # coe matrix is: (M + \lambdaI)^-1
        # M = np.dot(np.transpose(A), A)
        # l_i = l * np.identity(cols)
        # coe = - np.linalg.inv(M + l_i)

        # coe second approach is: (R^tR)^-1
        coe = - np.linalg.inv(np.dot(np.transpose(r), r))

    except:
        print "Step 0, problems"

    # Step 1, x_k = inv(R) * transpose(Q_1) * b
    try:
        inv_r = np.linalg.inv(r1)
        trans_q1 = np.transpose(q1)
        xk = np.dot(np.dot(inv_r, trans_q1), b)
    except:
        print "Step 1, problems"

    term = []
    term.append(to_array(xk))

    # Step 2, iteration
    if mode == 1:
        sk = xk
        for k in range(1, n + 1):
            sk = np.dot(coe, sk)
            tk = ma.pow(-1, k) * ma.pow(l, k) * sk
            xk = xk + tk
            term.append(to_array(xk))

    if mode == 2:
        tk = xk
        for k in range(1, n + 1):
            t = l * tk
            tk = np.dot(-coe, t)
            xk = xk + tk
            term.append(to_array(xk))

    #termasarray = np.asarray(term)

    #plot_vector(termasarray[:, 0], n+1)

    return xk


def svd_lambda(A, b, l):
    u, s, v = np.linalg.svd(A)
    u1 = u[0:rows, 0:rank]
    s1 = np.diag(s[0:rank])
    v1_t = v[0:rank, 0:cols]

    v1 = np.transpose(v1_t)
    s1_inv = np.linalg.inv(s1)
    u1_t = np.transpose(u1)

    if (l == 0):
        p1 = np.dot(v1, s1_inv)
        p2 = np.dot(p1, u1_t)
        x_svd = np.dot(p2, b)

    if (l != 0):
        l_i = l * np.identity(rank)
        t2 = np.linalg.inv(np.dot(s1, s1) + l_i)
        v1_t2 = np.dot(v1, t2)
        t3 = np.dot(s1, np.dot(u1_t, b))
        x_svd = np.dot(v1_t2, t3)

    return x_svd


def test():
    start = time.time()
    # Random Matrix
    A = createA(rows, cols, rank)
    l = lambda_svd(A, rank)
    b = np.random.rand(rows, 1)
    C = concatMatrix(A, ma.pow(l, 0.5))

    time_creation = time.time()

    # Solve using SVD l = 0
    x_svd_0 = svd_lambda(A, b, 0)
    time_svd_0 = time.time()

    # Solve using SVD l = lsvd
    x_svd_l = svd_lambda(A, b, l)
    time_svd_l = time.time()

    # Solve using iteration
    x_iteration = iteration(A, b, l, 3, 2)
    time_iteration = time.time()

    # Results
    print "Comparison"
    print "########################################################"
    print "x[0] SVD (l = 0): ", x_svd_0[0]
    print "x[0] SVD (l = lsvd): ", x_svd_l[0]
    print "x[0] iteration: ", x_iteration[0]
    print "########################################################"

    print ""
    print "Elapsed Time"
    print "########################################################"
    print "Example creation: ", time_creation - start
    print "Solve SVD (lambda = 0): ", time_svd_0 - time_creation
    print "Solve SVD (lambda = lsvd): ", time_svd_l - time_svd_0
    print "Iteration: ", time_iteration - time_svd_l
    print "########################################################"


def compactsvd(A):
    U, S, V = np.linalg.svd(A)
    U1 = U[0:rows, 0:rank]
    sigmak = S[rank - 1]
    S1 = np.diag(S[0:rank])
    V1_t = V[0:rank, 0:cols]
    V1 = V1_t.T
    return U1, S1, V1, sigmak


def xlambda(U1, S1, V1, b, l):
    t1 = l * np.identity(rank)
    t2 = np.linalg.inv(np.dot(S1, S1) + t1)
    t3 = np.dot(S1, np.dot(U1.T, b))
    x_lambda = np.dot(np.dot(V1, t2), t3)
    return x_lambda


def plot_x_svd(rows, cols, rank, betas):
    norms = []
    lambdas = []
    errors = []

    # for i in range(-steps, steps+1):
    #    lambda_i = i/(100000.0*steps)
    #    x_svd_l = svd_lambda(A, b, lambda_i)
    #    norms.append(np.linalg.norm(x_svd_l))
    #    lambdas.append(lambda_i)

    #plt.plot(lambdas, norms, 'ro')
    print "Creating matrix ..."
    A = createA(rows, cols, rank)
    b = np.random.rand(rows, 1)

    print "Calculating SVD ..."
    U1, S1, V1, sigmak = compactsvd(A)

    print "Getting x(lambda)"
    for beta in betas:
        l = beta * sigmak
        x_lambda = xlambda(U1, S1, V1, b, l)
        norms.append(np.linalg.norm(x_lambda))
        lambdas.append(l)
        errors.append(np.linalg.norm(np.dot(A, x_lambda) - b))

    plt.ylim(0, np.max(norms))
    plt.ylabel("norm(xlambda)")
    plt.xlabel("beta=sigmak2/lambda")
    plt.title("Norm vs beta, sigmak=%f" % sigmak)
    plt.plot(betas, norms, 'ro')
    plt.show()

    plt.ylim(np.min(errors), np.max(errors))
    plt.ylabel("norm(A*xlambda - b)")
    plt.xlabel("lambdas")
    plt.title("Errors vs lambdas")
    plt.plot(lambdas, errors, 'ro')
    plt.show()

#betas = np.arange(0,0.02,0.001)
#plot_x_svd(rows, cols, rank, betas)
test()
