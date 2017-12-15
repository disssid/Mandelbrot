import sys
import os
import psutil
import tempfile
from sys import stderr
from subprocess import Popen
from time import sleep

def cpu_stats():
    if not hasattr(psutil, "sensors_temperatures"):
        sys.exit("platform not supported")
    temps = psutil.sensors_temperatures()
    if not temps:
        sys.exit("can't read any temperature")
    for name, entries in temps.items():
        print(name)
        for entry in entries:
            print (
                entry.label or name, entry.current, entry.high,
                entry.critical)
        print()
    mem = psutil.virtual_memory()
    print(mem.available,mem.used)

def gpu_stats():
    temp_filename = tempfile.NamedTemporaryFile(mode='w')
    temp_filename.close()
    nvidia_tmp_filename = temp_filename.name

    command = 'nvidia-smi --query-gpu=pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,' \
              'memory.free,memory.used,power.draw,clocks.sm,clocks.mem --format=csv -f {}'.format(nvidia_tmp_filename)
    p = Popen(command, stdout=stderr, stderr=open(os.devnull, 'w'), shell=True)
    p.wait()
    output = []
    #sleep(1)
    with open(nvidia_tmp_filename, 'r') as f:
        output.extend(f.readlines()[1:])
    os.remove(nvidia_tmp_filename)
    #print("current,temperature,GPU utilization,Memory utilization," \
    #          "Available memory,Used memory,Power,Clocks SM,Clocks Memory")
    for o in output:
        print(o.replace(',',''))

if __name__ == '__main__':
    cpu_stats()
    gpu_stats()
