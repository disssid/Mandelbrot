from numba import jit, vectorize, guvectorize, float64, complex64, int32, float32
import numpy as np
import time
from stats import cpu_stats,gpu_stats
from time import sleep

@jit(int32(complex64, int32))
def mandelbrot(c,maxiter):
    nreal = 0
    real = 0
    imag = 0
    for n in range(maxiter):
        nreal = real*real - imag*imag + c.real
        imag = 2* real*imag + c.imag
        real = nreal;
        if real * real + imag * imag > 4.0:
            return n
    return 0

@guvectorize([(complex64[:], int32[:], int32[:])], '(n),()->(n)',target='parallel')
def mandelbrot_numpy(c, maxit, output):
    maxiter = maxit[0]
    for i in range(c.shape[0]):
        output[i] = mandelbrot(c[i],maxiter)
        
def mandelbrot_set(width,height,xmin,xmax,ymin,ymax,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    n3 = mandelbrot_numpy(c,maxiter)
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
