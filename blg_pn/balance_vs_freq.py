import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time
import math
import numpy as np
import labrad
import labrad.units as U
from include import CapacitanceBridge


## user inputs
data_dir = 'EMS062'
file_name = 'CREF_swap'

ref_atten = 90.
sample_atten = 0.
chY1 = 0.75

freq_range = [0.64,55.134]
num_points = 10


## start of code
freqs = np.logspace(np.log10(freq_range[0]),np.log10(freq_range[1]),num_points)

cxn = labrad.connect()
lck = cxn.sr830
lck.select_device(1)

acbox = cxn.acbox
acbox.select_device()

dv = cxn.data_vault

acbox.set_voltage('Y1',chY1)

bridge = CapacitanceBridge.CapacitanceBridgeSR830Lockin
s1 = np.array((0.5, 0.5)).reshape(2, 1)
s2 = np.array((-0.5, -0.5)).reshape(2, 1)


cb = bridge(lck=lck, acbox=acbox, time_const=0.1,
                         iterations=5, tolerance=5e-4,
                         s_in1=s1, s_in2=s2, excitation_channel='Y2')
ac_scale = 10**(-(ref_atten-sample_atten)/20.0)/chY1

try:
    dv.mkdir(data_dir)
    print "Folder {} was created".format(data_dir)
    dv.cd(data_dir)
except Exception:
    dv.cd(data_dir)


dv.new(file_name, ("j", "frequencies [Hz]"),
       ('Cs', 'Ds', 'C_', 'D_', 'phases [degrees]'))

for j in range(len(freqs)):
    lck.sensitivity(1.0)
    time.sleep(3.0)
    if freqs[j] < 12:
        cb = bridge(lck=lck, acbox=acbox, time_const=1.0,
                                 iterations=5, tolerance=5e-4,
                                 s_in1=s1, s_in2=s2, excitation_channel='Y2')

    else:
        cb = bridge(lck=lck, acbox=acbox, time_const=0.1,
                                 iterations=5, tolerance=5e-4,
                                 s_in1=s1, s_in2=s2, excitation_channel='Y2')


    acbox.set_frequency(freqs[j])
    print cb.balance()
    cs, ds = cb.capacitance(ac_scale)
    c_, d_ = cb.offBalance(ac_scale)
    phase = acbox.get_phase()
    dv.add(np.array([j, freqs[j], cs, ds, c_, d_, phase]).T)
