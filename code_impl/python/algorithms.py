import classes as cl
import pandas as pd
import numpy as np
from datetime import datetime, date, time
import time
import matplotlib as plt


def OVECMRR(y, L, P, it, avg_error, r, n, alpha):
    #it2, l = y.shape
    Y_true = np.zeros([it, l])
    Y_pred = np.zeros([it, l])
    mape = np.zeros([it, l])
    m = cl.Matrix()
    start_time = time.time()

    for i in range(it):
        #sys.stdout.write("\r{0}%".format((float(i + 1) / it) * 100))
        # sys.stdout.flush()
        y_i = y[i:i + L]
        if i == 0:
            beta = m.get_johansen(y_i.as_matrix(), P, r)
            A, B = m.vec_matrix(y_i, P, beta.evecr)
        else:
            A, B = m.vec_matrix_online(A, B, y_i, P, beta.evecr)
            Y_true[i, :] = B[-1,:]
            Y_pred[i, :] = np.dot(A[-1,:], x)

        x = m.ridge_regression(A, B, alpha)

        Ax = np.dot(A, x)
        residuals = B - Ax

        y_true = B[-n:]
        y_pred = Ax[-n:]

        mape[i, :], std = m.MAPE(y_true, y_pred)

        if (np.average(mape[i, :]) > avg_error):
            beta = m.get_johansen(y_i.as_matrix(), P)
            r = beta.r
            A = m.vec_matrix_update(A, y_i, P, beta.evecr)
            x = m.ridge_regression(A, B, alpha)

    o_mape, o_std = m.MAPE(Y_true, Y_pred)  # Out-of-Sample MAPE
    return y_true, y_pred, Y_true, Y_pred, mape, o_mape, o_std, time.time() - start_time


def OOVECM(y, L, P, it, r, n):
    #it2, l = y.shape

    Y_true = np.zeros([it, l])
    Y_pred = np.zeros([it, l])
    mape = np.zeros([it, l])

    m = cl.Matrix()
    start_time = time.time()
    for i in range(1, it):
        y_i = y[i:i + L]
        beta = m.get_johansen(y_i.as_matrix(), P, r)
        A, B = m.vec_matrix(y_i, P, beta.evecr)

        x, residuals, rank, s = np.linalg.lstsq(A, B)
        Ax = np.dot(A, x)
        residuals = B - Ax

        Y_true[i, :] = B[-1,:]
        Y_pred[i, :] = np.dot(A[-1,:], x)

    o_mape, o_std = m.MAPE(Y_true, Y_pred)
    return Y_true, Y_pred, o_mape, o_std, time.time() - start_time, residuals


def OVECM(y, L, P, it, avg_error, r, n):
    it2, l = y.shape

    dy_true = np.zeros([it, l], dtype=np.float32)
    dy_pred = np.zeros([it, l], dtype=np.float32)
    mape = np.zeros([it, l], dtype=np.float32)
    mae = np.zeros([it, l], dtype=np.float32)
    rmse = np.zeros([it, l], dtype=np.float32)

    m = cl.Matrix()

    start_time = time.time()

    for i in range(it):
        #sys.stdout.write("\r{0}%".format((float(i + 1) / it) * 100))
        # sys.stdout.flush()
        y_i = y[i:i + L]

        if i == 0:
            # Initialization
            beta = m.get_johansen(y_i.as_matrix(), P, r)
            A, B = m.vec_matrix(y_i, P, beta.evecr)
        else:
            # Update & Out-of-sample forecast
            A, B = m.vec_matrix_online(A, B, y_i, P, beta.evecr)
            dy_true[i-1, :] = B[-1,:]
            dy_pred[i-1, :] = np.dot(A[-1,:], x)

        # OLS
        x, residuals, rank, s = np.linalg.lstsq(A, B)
        Ax = np.dot(A, x)

        # Internal mape
        y_true = y_i.as_matrix()[-n:]
        y_pred = Ax[-n:] + y_i.as_matrix()[-n - 1:-1]

        # Stats info
        stats = cl.stats(y_true, y_pred)
        mape[i, :], mae[i,:], rmse[i,:] = stats.mape, stats.mae, stats.rmse
        avg_mape = np.average(mape[i, :])

        # Beta update
        if avg_mape > avg_error:
            beta = m.get_johansen(y_i.as_matrix(), P)
            r = beta.r
            A = m.vec_matrix_update(A, y_i, P, beta.evecr)
            x, residuals, rank, s = np.linalg.lstsq(A, B)

    # model_it object
    oecm = cl.model_it(
        y[L:L + it], dy_pred + y[L - 1:L + it - 1], dy_true, dy_pred, mape, mae, rmse)
    return oecm


def ECM(y, L, offset, P):
    m = cl.Matrix()
    start_time = time.time()

    L_max, n_var = y.shape
    if L + offset > L_max:
        print "L is bigger than L max"
        return

    y_i = y[offset: offset + L]

    beta = m.get_johansen(y_i.as_matrix(), P, 0)
    A, B = m.vec_matrix(y_i, P, beta.evecr)

    x, residuals, rank, s = np.linalg.lstsq(A, B)
    Ax = np.dot(A, x)
    residuals = B - Ax

    ecm = cl.ecm(y_i, Ax + y_i[P:-1], B, Ax, beta, P)

    return ecm


def akaike_cmp(residuals, nx, nlag):
    nobs, neqs = residuals.shape
    k = neqs * nlag + nx + 1
    T = nobs * neqs
    square = residuals ** 2
    RSS = np.sum(square)
    Loglikelihood = -(T / 2) * (1 + np.log(2 * np.pi) + np.log(RSS / T))
    AIC_m = -2 * Loglikelihood / T + 2 * k / T
    AIC_rss = T * np.log(RSS / T) + 2 * k

    return AIC_m, AIC_rss
