from __future__ import division

import numpy as np
import math as ma
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, date, time

import traceback
import sys
import random
import time

from statsmodels.tsa.johansen import coint_johansen
from statsmodels.tsa.stattools import adfuller
import glob

BETA = 1.0 / 9.0

ROWS = 3000
COLS = 1600
RANK = 632
R = 1
P = 3
FILES = ['EURUSD.csv']
size = 1

# invento paola 4


class model:

    def __init__(self, y, yhat, dy, dyhat):
        """Constructor model class
        """
        self.y = y
        self.yhat = yhat
        self.dy = dy
        self.dyhat = dyhat
        self.yresid = y - yhat
        self.dyresid = dy - dyhat


class ecm(model):

    def __init__(self, y, yhat, dy, dyhat, beta, nlag):
        """Constructor
        """
        model.__init__(self, y, yhat, dy, dyhat)

        self.beta = beta
        self.nobs, self.neqs = y.shape
        self.nobs = self.nobs - nlag - 1
        self.nlag = nlag
        self.nvar = self.neqs * self.nlag + self.beta.r + 1


class model_it(model):

    def __init__(self, y, yhat, dy, dyhat, mapes, maes, rmses):
        """Constructor
        """
        model.__init__(self, y, yhat, dy, dyhat)

        self.mapes = mapes
        self.maes = maes
        self.rmses = rmses


class stats:

    def __init__(self, y_true=None, y_pred=None):
        """Constructor"""
        try:
            self.mape = self.MAPE(y_true, y_pred)
            self.mae = self.MAE(y_true, y_pred)
            self.rmse = self.RMSE(y_true, y_pred)
        except:
            self.mape = 0
            self.mae = 0
            self.rmse = 0

    def filter(self,actual,forecast):
        indexes = actual.nonzero()   # checking zeros in actual
        actual = actual[indexes]
        forecast = forecast[indexes]

        indexes = forecast.nonzero() # checking zeros in forecast
        actual = actual[indexes]
        forecast = forecast[indexes]

        indexes = np.where(~np.isnan(forecast)) # checking nan in forecast
        actual = actual[indexes]
        forecast = forecast[indexes]

        return (actual,forecast)

    def MAPE(self, y_true, y_pred):
        """ Mean absolute percentage error
        """
        try:
            l = y_true.shape[1]
        except:
            y_true.shape += (1,)  # avoiding (l,) shape problem
            y_pred.shape += (1,)  # avoiding (l,) shape problem
            l = 1

        mape = np.zeros(l)

        for i in range(l):
            actual, forecast = self.filter(y_true[:,i],y_pred[:,i])
            mape[i] = np.mean(np.abs((actual - forecast) / actual)) * 100

        self.mape = mape
        return mape

    def MAE(self, y_true, y_pred):
        """Mean Average Error
        """
        try:
            l = y_true.shape[1]
        except:
            y_true.shape += (1,)  # avoiding (l,) shape problem
            y_pred.shape += (1,)  # avoiding (l,) shape problem
            l = 1

        mae = np.zeros(l)

        for i in range(l):
            actual, forecast = self.filter(y_true[:,i],y_pred[:,i])
            mae[i] = np.mean(np.abs(actual - forecast))

        self.mae = mae
        return mae

    def RMSE(self, y_true, y_pred):
        """Root Mean Square Error
        """
        try:
            l = y_true.shape[1]
        except:
            y_true.shape += (1,)  # avoiding (l,) shape problem
            y_pred.shape += (1,)  # avoiding (l,) shape problem
            l = 1

        rmse = np.zeros(l)

        for i in range(l):
            actual, forecast = self.filter(y_true[:,i],y_pred[:,i])
            rmse[i] = np.sqrt(np.mean((actual - forecast) ** 2))

        self.rmse = rmse
        return rmse

    def AIC(self, residuals, nx, nlag):
        """Akaike Information Criterion
        """
        nobs, neqs = residuals.shape
        k = neqs * nlag + nx + 1
        T = nobs * neqs
        square = residuals ** 2
        RSS = np.sum(square)
        Loglikelihood = -(T / 2) * (1 + np.log(2 * np.pi) + np.log(RSS / T))
        AIC_m = -2 * Loglikelihood / T + 2 * k / T
        AIC_rss = T * np.log(RSS / T) + 2 * k

        return AIC_m, AIC_rss


class Reader:

    def __init__(self, files=[]):
        """ data Constructor

        Parameters
        ----------
        filename: string
                  Data path
        """

        self.files = files

    def rename_columns(self, stock):
        """
        """
        tick_data = ['Ask', 'Bid', 'AskVolume', 'BidVolume']
        new_label = [stock + '_' + x for x in tick_data]

        return new_label

    def _mean(self, x):
        return np.mean(x)

    def load_dukascopy(self, files=None):
        """
        """
        if not files:
            files = self.files

        self.files = files

        for filename in files:
            df = pd.read_csv(
                filename, parse_dates={'Timestamp': ['Date', 'Time']}, index_col='Timestamp')
            df.columns = self.rename_columns(
                filename.split('/')[-1].split('.')[0])
            yield df

    def load(self):
        """
        """
        ticks_list = []

        ticks = self.load_dukascopy()

        for tick in ticks:
            ticks_list.append(tick)

        return ticks_list

    def resample(self, dataframes, freq='5s'):
        """ Ticks resampling at frequency freq

        dataframes: list of dataframes currency ticks
        freq:       resampling frequency (5 seconds default)
        """

        for df in dataframes:
            yield df.resample(freq, fill_method='bfill')

    def resample_list(self, ticks_list, freq, norm=None):
        """ Ticks resampling at frequency freq, merge dataframes and 
            resample nomalization (1/Currency)

        Parameters
        ----------
        ticks_list: list of dataframes currency ticks
        freq:       resampling frequency
        norm:       list of columns positions to normalize
        """

        ticksr_list = []
        ticks_resampled = self.resample(ticks_list, freq)

        for tickr in ticks_resampled:
            ticksr_list.append(tickr)

        ask, bid = self.merge(ticksr_list, 'Ask'), self.merge(
            ticksr_list, 'Bid')

        if norm:
            ask_norm, bid_norm = self.normalize(
                ask, norm), self.normalize(bid, norm)
            ask_norm, bid_norm = ask_norm.astype(
                np.float32), bid_norm.astype(np.float32)

            return ask_norm, bid_norm

        return ask.astype(np.float32), bid.astype(np.float32)

    def merge(self, dataframes, price):
        """
        """
        final = dataframes[0]
        offset = 4

        if price == 'Ask':
            index = 0
        elif price == 'Bid':
            index = 1
        elif price == 'Ask_Volume':
            index = 2
        elif price == 'Bid_Volume':
            index = 3

        cols = []
        cols.append(final.columns[index])

        for i in range(1, len(dataframes)):
            final = pd.merge(final, dataframes[
                             i], how='outer', left_index='Timestamp', right_index='Timestamp')
            cols.append(final.columns[i * offset + index])

        return final[cols]

    def normalize(self, dataframes, columns):
        """
        """
        n = dataframes.copy()
        new_columns = n.columns.tolist()

        for i in columns:
            n[n.columns[i]] = 1 / n[n.columns[i]]
            aux = n.columns[i]
            new_columns[i] = aux[3:6] + aux[0:3] + aux[6:]

        n.columns = new_columns

        return n

    def check_min_frequency(self, dataframes):
        """
        """
        #index = ask_norm[i].index[ask_norm[i].apply(np.isnan)]

        for df in dataframes:
            td = df.index.to_series().diff()
            max_diff = max(td)
            print df.columns[0:2], "Max gap between contiguous elements", max_diff / np.timedelta64(1, 's'), 'seconds'

    def sign_counter(true, pred):
        """
        """

        counter = 0
        if len(true) == len(pred):
            for i in range(len(true)):
                if np.sign(true[i]) == np.sign(pred[i]):
                    counter += 1
            return counter
        else:
            return -1


class Matrix:

    def __init__(self, rows=ROWS, cols=COLS, rank=RANK, r=R, lags=P, beta=BETA):
        """ matrix Constructor

        Parameters
        ----------
        rows: int
            Rows of the matrix
        cols: int
            Cols of the matrix
        rank: int
            Rank of the matrix
        r:
            cointegration
        lags:
            lags
        """

        self.rows = rows
        self.cols = cols
        self.rank = rank
        self.lags = lags
        self.r = R
        self.p = P
        self.beta = beta

    def ridge_regression(self, A, B, alpha=0):

        M = np.dot(A.T, A)
        N = np.linalg.inv(M + alpha * np.eye(A.shape[1]))
        P = np.dot(N, A.T)
        X = np.dot(P, B)

        return X

    def get_johansen(self, y, p, r=0):
        """
        Get the cointegration vectors at 95% level of significance
        given by the trace statistic test.
        """

        N, l = y.shape
        jres = coint_johansen(y, 0, p)
        trstat = jres.lr1                       # trace statistic
        tsignf = jres.cvt                       # critical values

        if not r:
            for i in range(l):
                if trstat[i] > tsignf[i, 1]:     # 0: 90%  1:95% 2: 99%
                    r = i + 1
                if np.sign(jres.evec[0, i]) == -1:  # sign of first elem
                    jres.evec[:, i] = -jres.evec[:, i]

        jres.r = r
        jres.evecr = jres.evec[:, :r]

        return jres

    def vec_matrix(self, y, p, evecr):
        """ Create vector error correction (VEC) matrix
        using time series differeces and
        johansen cointegration vectors.
        System Ax = B

        Parameters
        ----------
        y: pandas DataFrame
            prices time series of differents assets
        p: int
            number of lags of VEC model
        evecr: numpy.ndarray
            cointegration vectors

        Returns
        -------
        A: numpy.ndarray
            Vec matrix A
        B: numpy.ndarray
            Differences matrix B"""

        # N: n of prices, l: n of time series
        N, l = y.shape

        # First differences time series
        dy = y.diff(1)
        B = dy[p + 1:]
        new_col = dy
        A = new_col

        for i in range(p - 1):
            # Differences column shifted
            new_col = new_col.shift(1)

            # Adding differences columns
            A = [new_col, A]
            A = pd.concat(A, axis=1)                # Concatenate columns

        y = np.matrix(y)

        if evecr.size:
            new_cols = y * evecr
            new_cols = pd.DataFrame(new_cols, index=A.index)
            A = [A, new_cols]
            A = pd.concat(A, axis=1)

        A['ones'] = np.ones(N)              # Adding columns with 1's

        A = A[p:-1]                         # Removing first p rows

        A = A.as_matrix()
        B = B.as_matrix()
        return (A, B)

    def vec_matrix_online(self, A, B, y, P, beta):
        """ Create vector error correction online (VEC) matrix
        using time series differeces and johansen cointegration
        vectors. This method only change a row of A and B.
        System Ax = B

        Parameters
        ----------
        A: numpy.ndarray
            Old VEC matrix
        B: numpy.ndarray
            Old Differences matrix
        y: pandas DataFrame
            prices time series of differents assets
        P: int
            Lags
        beta: coint object
            Cointegrations vectors

        Returns
        -------
        A: numpy.ndarray
           Design Matrix
        B: Numpy.ndarray
           Matrix of responses """

        y_p = y[-(P + 2):]
        new_rowA, new_rowB = self.vec_matrix(y_p, P, beta)
        A = np.delete(A, (0), axis=0)  # Delete first row
        A = np.vstack((A, new_rowA))

        B = np.delete(B, (0), axis=0)  # Delete first row
        B = np.vstack((B, new_rowB))

        return A, B

    def vec_matrix_update(self, A, y, P, evecr):
        """ Update vector error correction (VEC) matrix
        using time series differeces and
        johansen cointegration vectors.
        System Ax = B

        Parameters
        ----------
        A: numpy.ndarray
            VEC Matrix to update
        y: pandas DataFrame
            prices time series of differents assets
        P: int
            number of lags of VEC model
        evecr: numpy.ndarray
            cointegration vectors

        Returns
        -------
        A: numpy.ndarray
            Vec matrix A """

        l = y.shape[1]
        A_old = A[:, :(P) * l]

        y = np.matrix(y)

        if evecr.size:
            new_cols = y * evecr
            A_old = np.concatenate((A_old, new_cols[P:-1]), axis=1)

        N = A.shape[0]
        A_old = np.concatenate((A_old, np.ones([N, 1])), axis=1)

        return np.array(A_old)

    def ADF(self, v, crit='5%', max_d=6, reg='nc', autolag='AIC'):
        """ Augmented Dickey Fuller test

        Parameters
        ----------
        v: ndarray matrix
            residuals matrix

        Returns
        -------
        bool: boolean
            true if v pass the test
        """

        boolean = True
        try:
            l = v.shape[1]
            for j in range(l):
                adf = adfuller(v[:, j], max_d, reg, autolag)

                if(adf[0] < adf[4][crit]):
                    pass
                else:
                    boolean = False
                    break
        except:
            adf = adfuller(v, max_d, reg, autolag)
            if(adf[0] > adf[4][crit]):
                boolean = False

        return boolean

    def MAPE(self, y_true, y_pred):
        """ Mean absolute percentage error

        Parameters
        ----------
        y_true:
        y_pred:

        Return
        MAPE: Mean absolute percentage error
        """
        try:
            l = y_true.shape[1]
        except:
            y_true.shape += (1,)  # avoiding (l,) shape problem
            y_pred.shape += (1,)  # avoiding (l,) shape problem
            l = 1

        mape = np.zeros(l)
        std = np.zeros(l)

        for i in range(l):
            indexes = y_true[:, i].nonzero()
            actual = y_true[:, i][indexes]
            forecast = y_pred[:, i][indexes]
            mape[i] = np.mean(np.abs((actual - forecast) / actual)) * 100
            std[i] = np.std(np.abs((actual - forecast) / actual) * 100)

        return mape, std

    def var_matrix(self, y, p):
        """ Create vector autorregresive (VAR) matrix

        Parameters
        ----------
        y: pandas DataFrame
           prices time series of differents assets
        p: int
           number of lags of VAR model

        Returns
        -------
        A: pandas DataFrame
           Design Matrix
        B: pandas DataFrame
           Matrix of responses """

        N, l = y.shape            # N: n of prices, l: n of time series
        new_col = y
        A = new_col

        for i in range(p - 1):
            new_col = new_col.shift(1)             # Column shifted
            A = [new_col, A]                       # Adding columns
            A = pd.concat(A, axis=1)               # Concatenate columns

        A['ones'] = np.ones(N)                     # Adding columns with 1's
        # Removing first p-1 rows (with Nan values)
        A = A[p - 1: -1]
        B = y[p:]

        return (A, B)

    def randomMatrix(self, rows=None, cols=None, rank=None):
        """ Create random matrix
        Parameters
        ----------
        rows: int
            number of rows of the matrix
        cols: int
            number of cols of the matrix
        rank: int
            rank of the matrix
        Returns
        -------
        new_A: ndarray
            random matrix created """

        if not rows:
            rows = self.rows

        if not cols:
            cols = self.cols

        if not rank:
            rank = self.rank

        A = np.random.rand(rows, cols)
        u, s, v = np.linalg.svd(A)

        S = np.zeros((rows, cols))

        new_sigma = np.random.uniform(0.001, 0.1, rank)
        new_sigma.sort()
        new_sigma = size * new_sigma[::-1]
        S[:self.rank, :self.rank] = np.diag(new_sigma)

        new_A = np.dot(u, np.dot(S, v))

        return new_A

    def to_array(self, v):
        vector = []
        for i in v:
            vector.append(i[0])

        return np.asarray(vector)

    def lambda_svd(self, A, rank=None):
        """ Lambda SVD
        Parameters
        ----------
        rank: int
            rank of the matrix A

        Returns
        -------
        lsvd: double
            lambda estimation using beta """

        if not rank:
            rank = self.rank

        u, s, d = np.linalg.svd(A)

        lsvd = self.beta * s[rank - 1] * s[rank - 1]

        return lsvd

    def sigma_svd(self, A, rank=None):
        """ Lambda SVD
        Parameters
        ----------
        rank: int
            rank of the matrix A

        Returns
        -------
        sigmasvd: double
            smallest sigma from svd """

        if not rank:
            rank = self.rank

        u, s, d = np.linalg.svd(A)

        sigmasvd = s[rank - 1]

        return sigmasvd

    def compactsvd(self, A, rows=None, cols=None, rank=None):
        """ Compact SVD
        Parameters
        ----------
        A: ndarray
            Matrix

        Returns
        -------
        U1: ndarray
            Unitarty matrix
        S1: ndarray
            Diagonal matrix
        V1: ndarray
            Unitary matrix
        sigmak: double
            smallest sigma from svd """

        if not rows:
            rows = self.rows

        if not cols:
            cols = self.cols

        if not rank:
            rank = self.rank

        U, S, V = np.linalg.svd(A)
        U1 = U[0:rows, 0:rank]
        sigmak = S[rank - 1]
        S1 = np.diag(S[0:rank])
        V1_t = V[0:rank, 0:cols]
        V1 = V1_t.T

        return U1, S1, V1, sigmak

    def xlambda(self, U1, S1, V1, b, l):
        """ X lambda
        Parameters
        ----------
        U1: ndarray
            Unitary matrix
        S1: ndarray
            Diagonal matrix
        V1: ndarray
            Unitary matrix
        b: ndarray
            Columbn vector from linear system
        l: double
            lambda

        Returns
        -------
        x_lambda: ndarray
            least square solution """

        t1 = l * np.identity(rank)
        t2 = np.linalg.inv(np.dot(S1, S1) + t1)
        t3 = np.dot(S1, np.dot(U1.T, b))
        x_lambda = np.dot(np.dot(V1, t2), t3)
        return x_lambda

    def concatMatrix(self, a, l, cols):
        """ Concatenate matrix a with lambda times identity
        Parameters
        ----------
        a: numpy.ndarray
            Matrix
        l: double
            Lambda

        Returns
        -------
        concat: numpy.ndarray
            Concatenated Matrix """
        l_i = l * np.identity(cols)
        concat = np.concatenate((a, l_i))

        return concat

    def lambda_qr(self, A, factor=None):
        """ Calculate the lambda_qr Coleman Paper

        Parameters
        ----------
        A: numpy.ndarray
            Matrix to calculate to lambda qr

        Returns
        -------
        l_qr: double
            Lambda QR """
        if not factor:
            factor = 0.00025

        q1, r1 = np.linalg.qr(A)

        w = np.abs(q1.diagonal())
        w_min = np.amin(w)
        w_max = np.amax(w)
        l_1 = factor * w_min ** 2
        l_2 = l_1 / (w_max ** 2)
        l_qr = 0.5 * (l_1 + l_2)

        return l_qr

    def iteration(self, A, b, n, mode, l):
        """ Iteration method is an alternative to OLS. It has 
        2 modes, and use A (coefficients), B (constants), 
        lambda value and n iterations
        Parameters
        ----------
        A: numpy.ndarray
            Coefficients matrix
        b: numpy.ndarray
            Constans matrix
        l: double
            Lambda
        n: int
            Iterations
        mode: int
            Determine the mode to use

        Returns
        -------
        xk: numpy.ndarray
            Aproximated solution """
        # Step 0, qr factorization of C, and create Q_1, -inverse(rows+lI)
        rows, cols = A.shape

        C = self.concatMatrix(A, ma.pow(l, 0.5), cols)
        q, r = np.linalg.qr(C)
        q1 = q[0:rows, 0:cols]
        r1 = r[0:rows, 0:cols]

        # coe second approach is: (R^tR)^-1
        coe = - np.linalg.inv(np.dot(np.transpose(r), r))

        # Step 1, x_k = inv(R) * transpose(Q_1) * b
        inv_r = np.linalg.inv(r1)
        trans_q1 = np.transpose(q1)
        xk = np.dot(np.dot(inv_r, trans_q1), b)
        term = []
        term.append(self.to_array(xk))

        # Step 2, iteration
        if mode == 1:
            sk = xk
            for k in range(1, n + 1):
                sk = np.dot(coe, sk)
                tk = ma.pow(-1, k) * ma.pow(l, k) * sk
                xk = xk + tk
                term.append(self.to_array(xk))

        if mode == 2:
            tk = xk
            for k in range(1, n + 1):
                t = l * tk
                tk = np.dot(-coe, t)
                xk = xk + tk
                term.append(self.to_array(xk))

        return xk


class Util:

    def __init__(self):
        self.general = "General"

    def plot_vector(self, v, n, xlabel, ylabel, title):
        plt.plot(range(1, n + 1), v, 'ro')
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)

        plt.show()
