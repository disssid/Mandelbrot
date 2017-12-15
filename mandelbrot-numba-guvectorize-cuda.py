from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import numpy as np
import time
from stats import cpu_stats,gpu_stats
from time import sleep			

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),(n)->(n)', target='cuda')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        creal = c[i].real
        cimag = c[i].imag
        real = creal
        imag = cimag
        output[i] = 0
        for n in range(maxiter):
            real2 = real*real
            imag2 = imag*imag
            if real2 + imag2 > 4.0:
                output[i] = n
                break
            imag = 2* real*imag + cimag
            real = real2 - imag2 + creal
            
        
def mandelbrot_set(width,height,xmin,xmax,ymin,ymax,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = np.empty(c.shape, int)
    maxit = np.ones(c.shape, int) * maxiter
    n3 = mandelbrot_numpy(c,maxit)
    return (r1,r2,n3.T) 
cpu_stats()
gpu_stats()
start_time = time.time()
mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,200)
print("Mandelbrot set 1 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,400)
print("Mandelbrot set 1 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,600)
print("Mandelbrot set 1 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,800)
print("Mandelbrot set 1 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-2.0,0.5,-1.25,1.25,1000)
print("Mandelbrot set 1 --- %s seconds ---" % (time.time() - start_time))
cpu_stats()
gpu_stats()
sleep(2)

print("SET 2 params 1000,1000,-0.74877,-0.74872,0.06505,0.06510")
cpu_stats()
gpu_stats()
start_time = time.time()
mandelbrot_set(1000,1000,-0.74877,-0.74872,0.06505,0.06510,200)
print("Mandelbrot set 2 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-0.74877,-0.74872,0.06505,0.06510,400)
print("Mandelbrot set 2 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-0.74877,-0.74872,0.06505,0.06510,600)
print("Mandelbrot set 2 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-0.74877,-0.74872,0.06505,0.06510,800)
print("Mandelbrot set 2 --- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
mandelbrot_set(1000,1000,-0.74877,-0.74872,0.06505,0.06510,1000)
print("Mandelbrot set 2 --- %s seconds ---" % (time.time() - start_time))
cpu_stats()
gpu_stats()
