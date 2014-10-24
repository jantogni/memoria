import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np

import scikits.cuda.linalg as culinalg
import scikits.cuda.misc as cumisc
culinalg.init()

import string
import scikits.cuda.cula as cula

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


def cuda_qr():
    cula.culaInitialize()
    
    a=np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float32)
    
    n = a.shape[0]
    m = a.shape[1]
    ida = a.shape[1]
    
    tau = np.empty(n,dtype=np.int32)
    
    tau_gpu = gpuarray.to_gpu(tau)
    a_gpu = gpuarray.to_gpu(a)
    
    #culaDeviceDgeqrf
    output = cula.culaDeviceSgeqrf(m, n, a_gpu.gpudata, ida, tau_gpu.gpudata)
    
    print a_gpu.get()
    print tau_gpu.get()
    
    cula.culaShutdown()


def cuda_svd():
    #demo_types = [np.float32, np.complex64]
    demo_types = [np.float32]
    
    if cula._libcula_toolkit == 'premium' and \
           cumisc.get_compute_capability(pycuda.autoinit.device) >= 1.3:
        demo_types.extend([np.float64, np.complex128])
    
    for t in demo_types:
        print 'Testing svd for type ' + str(np.dtype(t))
    
        #numpy.float32
        a = np.asarray((np.random.rand(1000, 1000)-0.5)/10, t)
    
        print a.shape

        u_cpu, s_cpu, v_cpu = np.linalg.svd(a)    
    
        #gpu array
        a_gpu = gpuarray.to_gpu(a)
    
        #call cula rutine
        u_gpu, s_gpu, vh_gpu = culinalg.svd(a_gpu)
    
        print u_gpu.get()
        print u_cpu    
    
        #print s_gpu.get()
        #print vh_gpu.get()
    
        a_rec = np.dot(u_gpu.get(), np.dot(np.diag(s_gpu.get()), vh_gpu.get()))
        print 'Success status: ', np.allclose(a, a_rec, atol=1e-3)
        print 'Maximum error: ', np.max(np.abs(a-a_rec))
        print ''


@do_profile(follow=[cuda_svd, cuda_qr])
def main():
    cuda_svd()
    cuda_qr()

if __name__ == "__main__":
    main()

