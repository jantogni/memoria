import classes as cl
import numpy as np
import time
import sys
from statsmodels.tsa.stattools import adfuller
import algorithms as alg


try:
    from line_profiler import LineProfiler

    def do_profile(follow=[]):
        def inner(func):
            def profiled_func(*args, **kwargs):
                try:
                    profiler = LineProfiler()
                    profiler.add_function(func)
                    for f in follow:
                        profiler.add_function(f)
                    profiler.enable_by_count()
                    return func(*args, **kwargs)
                finally:
                    profiler.print_stats()
            return profiled_func
        return inner

except ImportError:
    def do_profile(follow=[]):
        "Helpful if you accidentally leave in production!"
        def inner(func):
            def nothing(*args, **kwargs):
                return func(*args, **kwargs)
            return nothing
        return inner


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
    oecm = cl.model_it(y[L:L + it], dy_pred + y[L - 1:L + it - 1], dy_true, dy_pred, mape, mae, rmse)
    return oecm


def OVECMRR(y, L, P, it, avg_error, r, n, alpha):
    l = y.shape[1]
    new_beta = False            # Get a new beta condition
    Y_true = np.zeros([it, l])  # Out of sample true value
    Y_pred = np.zeros([it, l])  # Out of sample forecasting
    mape = np.zeros([it, l])     # In-sample mape
    m = cl.Matrix()

    print "OVECMRR loop"
    start_time = time.time()

    for i in range(it):
        sys.stdout.write("\r{0}%".format((float(i + 1) / it) * 100))
        sys.stdout.flush()

        y_i = y[i:i + L]

        if i == 0:
            beta = m.get_johansen(y_i.as_matrix(), P, r)
            A, B = m.vec_matrix(y_i, P, beta.evecr)        
        else:
            A, B = m.vec_matrix_online(A, B, y_i, P, beta.evecr)        
            Y_true[i,:] = B[-1,:]
            Y_pred[i,:] = np.dot(A[-1,:], x)

        x = m.ridge_regression(A, B, alpha)

        Ax = np.dot(A, x)
        residuals = B - Ax

        y_true = B[-n:]
        y_pred = Ax[-n:]

        mape[i,:] = m.MAPE(y_true, y_pred)

        if (np.average(mape[i,:]) > avg_error):
            beta = m.get_johansen(y_i.as_matrix(), P)
            r = beta.r
            A = m.vec_matrix_update(A, y_i, P, beta.evecr)
            x = m.ridge_regression(A, B, alpha)

    o_mape = m.MAPE(Y_true, Y_pred)  # Out-of-Sample MAPE
    return y_true, y_pred, Y_true, Y_pred, mape, o_mape, time.time() - start_time

def tests(data):
    m = cl.Matrix()
    y = np.log(data)                 # Logarithm of currencie
    r = 0
    Ls = [300, 3000]
    Ps = [4, 8]
    its = [10000]
    ns = [100]
    avg_errors = [0, 110, 120]
    f = open('tests.csv', 'a')
    f.write('r,L,P,it,n,avg_error,ttime,avgtime,MAPE EURUSD,MAPE GBPUSD, MAPE CHFUSD, MAPE JPYUSD \n')
    f.close()
    for L in Ls:
        for P in Ps:
            for it in its:
                # llamar a SLECM
                # imprimir a archivo
                for n in ns:
                    for avg_error in avg_errors:
                        print "running for: L: ", L, " P: ", P, " it: ", it, " n: ", n, " avg_error: ", avg_error 
                        y_true, y_pred, Y_true, Y_pred, mape, o_mape, etime = OVECM(y, L, P, it, avg_error, r, n)
                        f = open('tests.csv', 'a')
                        f.write(str(r)+','+str(L)+','+str(P)+','+str(it)+','\
                                + str(n)+','+str(avg_error)+','+str(etime)+','+str(etime/it)+","+','.join(map(str, o_mape))+"\n")
                        f.close()
    
@do_profile(follow=[OVECM, OVECMRR])
def main():
    path = '../data_csv/data_ticks/august_11/'
    assets = ['EURUSD','GBPUSD','USDCHF','USDJPY']
    
    data_list = [path + i + '.csv' for i in assets]
    
    reader = cl.Reader(data_list)
    ticks_list = reader.load()

    ask_norm, bid_norm = reader.resample_list(ticks_list, '60s', [2, 3])

    y = ask_norm

    L = 1000
    P = 4
    it = 400
    r = 0
    n = 50
    avg_error = 0.002

    oecm = OVECM(y, L, P, it, avg_error, r, n)

def synthetic_main():

    tickers = ['col1', 'col2', 'col3', 'col4']  
    print "Reading data ..."
    reader = cl.Reader('../data_csv/synthetic-data.csv')
    data = reader.load_data(tickers)
    data = data - data.min()  # Avoiding negative numbers


    y = np.log(data)                 # Logarithm of currencies
    L = 3000                    # Length of window
    P = 4                       # Number of lags
    it, l = y.shape
    it = it - L
    it = 100
    it = 10000
    r = 1                       # Number of cointegration relations, if 0 get Johansen r
    n = 50                      # Number of instances to calculate MAPE
    avg_error = 0             # MAPE threshold to get new beta
    y_true, y_pred, Y_true, Y_pred, mape, o_mape, etime = OVECM(y, L, P, it, avg_error, r, n)
    print "elapsed time: ", etime 
    print "Out-of-sample MAPE", o_mape


if __name__ == "__main__":
    main()
    # synthetic_main()
