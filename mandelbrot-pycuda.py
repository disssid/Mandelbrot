import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel
import time
import numpy as np

complex_gpu = ElementwiseKernel(
    "pycuda::complex<float> *q, int *output, int maxiter",
    """
    {
        float nreal, real = 0;
        float imag = 0;
        output[i] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            float real2 = real*real;
            float imag2 = imag*imag;
            nreal = real2 - imag2 + q[i].real();
            imag = 2* real*imag + q[i].imag();
            real = nreal;
            if (real2 + imag2 > 4.0f){
                output[i] = curiter;
                break;
                };
        };
    }
    """,
    "complex5",
    preamble="#include <pycuda-complex.hpp>",)

def mandelbrot_gpu(c, maxiter):
    q_gpu = gpuarray.to_gpu(c.astype(np.complex64))
    iterations_gpu = gpuarray.to_gpu(np.empty(c.shape, dtype=np.int))
    complex_gpu(q_gpu, iterations_gpu, maxiter)

    return iterations_gpu.get()

def mandelbrot_set(width,height,xmin,xmax,ymin,ymax,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    n3 = mandelbrot_gpu(c,maxiter)
    n3 = n3.reshape((width,height))
    return (r1,r2,n3.T)

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

print("SET 2 params 1000,1000,-0.74877,-0.74872,0.06505,0.06510")
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
