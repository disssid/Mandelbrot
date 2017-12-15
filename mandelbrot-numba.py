from __future__ import print_function, division, absolute_import

from timeit import default_timer as timer
from matplotlib.pylab import imshow, jet, show, ion
import matplotlib.pyplot as plt
import numpy as np
from stats import cpu_stats,gpu_stats
from time import sleep
from numba import jit


@jit
def mandel(x, y, max_iters):

    i = 0
    c = complex(x,y)
    z = 0.0j
    for i in range(max_iters):
        z = z*z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return 255

@jit
def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color

    return image


image = np.zeros((500 * 2, 500 * 2), dtype=np.uint8)
cpu_stats()
gpu_stats()
print("Mandelbrot SET1")
s = timer()
create_fractal(-2.0, 0.5, -1.25, 1.25, image, 200)
e = timer()
print(e - s)
plt.imsave('output1.png',image)
s = timer()
create_fractal(-2.0, 0.5, -1.25, 1.25, image, 400)
e = timer()
print(e - s)
plt.imsave('output2.png',image)
s = timer()
create_fractal(-2.0, 0.5, -1.25, 1.25, image, 600)
e = timer()
print(e - s)
plt.imsave('output3.png',image)
s = timer()
create_fractal(-2.0, 0.5, -1.25, 1.25, image, 800)
e = timer()
print(e - s)
plt.imsave('output4.png',image)
s = timer()
create_fractal(-2.0, 0.5, -1.25, 1.25, image, 1000)
e = timer()
print(e - s)
plt.imsave('output5.png',image)

cpu_stats()
gpu_stats()

sleep(2)
print("Mandelbrot SET2")
cpu_stats()
gpu_stats()
s = timer()
create_fractal(-0.74877,-0.74872,0.06505,0.06510, image, 200)
e = timer()
print(e - s)
plt.imsave('output6.png',image)
s = timer()
create_fractal(-0.74877,-0.74872,0.06505,0.06510, image, 400)
e = timer()
print(e - s)
plt.imsave('output7.png',image)
s = timer()
create_fractal(-0.74877,-0.74872,0.06505,0.06510, image, 600)
e = timer()
print(e - s)
plt.imsave('output8.png',image)
s = timer()
create_fractal(-0.74877,-0.74872,0.06505,0.06510, image, 800)
e = timer()
print(e - s)
plt.imsave('output9.png',image)
s = timer()
create_fractal(-0.74877,-0.74872,0.06505,0.06510, image, 1000)
e = timer()
print(e - s)
cpu_stats()
gpu_stats()
plt.imsave('output10.png',image)
