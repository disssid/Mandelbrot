from numba import vectorize, complex64, boolean, jit
import time
import numpy as np
from stats import cpu_stats,gpu_stats
from time import sleep	

@vectorize([boolean(complex64)])
def f(z):
    return (z.real*z.real + z.imag*z.imag) < 4.0

@vectorize([complex64(complex64, complex64)])
def g(z,c):
    return z*z + c 

@jit
def mandelbrot_numpy(c, maxiter):
    output = np.zeros(c.shape, np.int)
    z = np.empty(c.shape, np.complex64)
    for it in range(maxiter):
        notdone = f(z)
        output[notdone] = it
        z[notdone] = g(z[notdone],c[notdone]) 
    output[output == maxiter-1] = 0
    return output

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
