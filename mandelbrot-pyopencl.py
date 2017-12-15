import pyopencl as cl
import time
import numpy as np
from stats import cpu_stats,gpu_stats
from time import sleep

ctx = cl.create_some_context(interactive=True)
 
def mandelbrot_gpu(q, maxiter):

    global ctx
    
    queue = cl.CommandQueue(ctx)
    
    output = np.empty(q.shape, dtype=np.uint16)

    prg = cl.Program(ctx, """#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q,
                     __global ushort *output, ushort const maxiter)
    {
        int gid = get_global_id(0);
        float real = q[gid].x;
        float imag = q[gid].y;
        output[gid] = 0;
        for(int curiter = 0; curiter < maxiter; curiter++) {
            float real2 = real*real, imag2 = imag*imag;
            if (real*real + imag*imag > 4.0f){
                 output[gid] = curiter;
                 return;
            }
            imag = 2* real*imag + q[gid].y;
            real = real2 - imag2 + q[gid].x;
            
        }
    }
    """).build()

    mf = cl.mem_flags
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)


    prg.mandelbrot(queue, output.shape, None, q_opencl,
                   output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()
    
    return output

def mandelbrot_set(width,height,xmin,xmax,ymin,ymax,maxiter):
    r1 = np.linspace(xmin, xmax, width, dtype=np.float32)
    r2 = np.linspace(ymin, ymax, height, dtype=np.float32)
    c = r1 + r2[:,None]*1j
    c = np.ravel(c)
    n3 = mandelbrot_gpu(c,maxiter)
    n3 = n3.reshape((width,height))
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
