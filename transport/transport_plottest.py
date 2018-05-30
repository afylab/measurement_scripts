'''
[info]
version = 2.0
'''
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import time
import math
import numpy as np
import labrad
import labrad.units as U
import yaml


ADC_CONVERSIONTIME = 250
ADC_AVGSIZE = 1

adc_offset = np.array([0.29391179, 0.32467712])
adc_slope = np.array([1.0, 1.0])

def vs_fixed(p0, n0, delta, vs):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vs: fixed voltage set on graphene sample
    :return: (v_top, v_bottom)
    """
    return vs + 0.5 * (n0 + p0) / (1.0 + delta), vs + 0.5 * (n0 - p0) / (1.0 - delta)

def vs_fixed_1gate(p0,n0,delta,vs):
    """
    :param p0: polarizing field
    :param n0: charge carrier density
    :param delta: capacitor asymmetry
    :param vs: fixed voltage set on graphene sample
    :return: (v_gate, 0)
    """
    return n0, 0.0*n0


def function_select(s):
    """
    :param s: # of gates
    :return: function f
    """
    if s == 1:
        f = vs_fixed_1gate
    elif s == 2:
        f = vs_fixed
    return f


def lockin_select(cfg,cxn):
    lck = cxn.sr830
    lockins_requested = cfg['measurement']['lockins']
    lockins_present = len(lck.list_devices())
    if lockins_present<lockins_requested:
        sys.exit(['Not enough lockins connected to perform measurement.'])

    if cfg['measurement']['autodetect_lockins']:
        if (lockins_present==1)&(lockins_requested==1):
            return 0,0
        elif (lockins_present==2)&(lockins_requested==2):
            lck.select_device(0)
            if lck.input_mode()<2:
                if(cfg['lockin1']['type'])=='V':
                    return 0,1
                elif(cfg['lockin1']['type'])=='I':
                    return 1,0
        elif lockins_requested==1:
            for i in range(lockins_present):
                lck.select_device(i)
                if ((cfg['lockin1']['type'])=='V')&(lck.input_mode()<2):
                    return i,1
                elif ((cfg['lockin1']['type'])=='I')&(lck.input_mode()>=2):
                    return i,1
        else:
            sys.ext(['Lockin autodetection failed.'])
    else:
        lck_list = lck.list_devices()
        lck_num = []
        l1= -1
        l2 = -1
        for j in range(len(lck_list)):
            lck_num = lck_list[j][1]
            lck_num = lck_num.split("::")[-2]
            lck_num = int(lck_num)
            print lck_num
            if int(cfg['lockin1']['GPIB']) == lck_num:
                l1 = j
            elif (lockins_requested==2)&(int(cfg['lockin2']['GPIB']) == lck_num):
                l2 = j

        if (l1==-1)|((lockins_requested==2)&(l2==-1)):
			sys.exit(['Lockins not found, please check GPIB addresses.'])
        return l1,l2

def reshape_data(data,mult):
    """
    this bins and averages the output of the ADC when a higher point density (mult*# of data points)
    is used to make the gate sweep more smooth
    """
    data_reshaped = np.reshape(data,(np.shape(data)[0],np.shape(data)[1]/mult,mult))
    data_reshaped = np.mean(data_reshaped,2)
    return data_reshaped

def mesh(vfixed, offset, drange, nrange, gates=1, pxsize=(100, 100), delta=0.0):
    """
    drange and nrange are tuples (dmin, dmax) and (nmin, nmax)
    offset  is a tuple of offsets:  (N0, D0)
    pxsize  is a tuple of # of steps:  (N steps, D steps)
    fixed sets the fixed channel: "vb", "vt", "vs"
    fast  - fast axis "D" or "N"
    """
    f = function_select(gates)
    p0 = np.linspace(drange[0], drange[1], pxsize[1]) - offset[1]
    n0 = np.linspace(nrange[0], nrange[1], pxsize[0]) - offset[0]
    n0, p0 = np.meshgrid(n0, p0)  # p0 - slow n0 - fast
    # p0, n0 = np.meshgrid(p0, n0)  # p0 - slow n0 - fast
    v_fast, v_slow = f(p0, n0, delta, vfixed)
    return np.dstack((v_fast, v_slow)), np.dstack((p0, n0))

def create_file(dv, cfg, **kwargs): # try kwarging the vfixed
    try:
        dv.mkdir(cfg['file']['data_dir'])
        print "Folder {} was created".format(cfg['file']['data_dir'])
        dv.cd(cfg['file']['data_dir'])
    except Exception:
        dv.cd(cfg['file']['data_dir'])

    #measurement = cfg['measurement']

    gate1 = cfg['gate1']['type']
    if cfg['measurement']['gates'] == 1:
        gate2 = 'none'
    else:
        gate2 = cfg['gate2']['type']

    plot_parameters = {'extent': [cfg['meas_parameters']['n0_rng'][0],
                                  cfg['meas_parameters']['n0_rng'][1],
                                  cfg['meas_parameters']['p0_rng'][0],
                                  cfg['meas_parameters']['p0_rng'][1]],
                       'pxsize': [cfg['meas_parameters']['n0_pnts'],
                                  cfg['meas_parameters']['p0_pnts']]
                      }

    dv.new(cfg['file']['file_name']+"-plot", ("i", "j", gate1, gate2),
           ('Ix', 'Iy', 'Vx', 'Vy', 'p0', 'n0', 'R', 'sigma', 't'))
    print("Created {}".format(dv.get_name()))
    dv.add_comment(cfg['file']['comment'])


    #parameters for "grapher", giving the columns and labels for two data plots, and the "x" variable
    dv.add_parameter('data1_col', 10)
    dv.add_parameter('data1_label', 'R [Ohm]')
    dv.add_parameter('data2_col', 11)
    dv.add_parameter('data2_label', 'sigma [1/Ohm]')
    dv.add_parameter('x_col',9)
    dv.add_parameter('extent', tuple(plot_parameters['extent']))
    dv.add_parameter('pxsize', tuple(plot_parameters['pxsize']))

    """ parameters for data_vault_plotter
    ranges of x and y variables for data_vault_plotter, which take
    the form of "x_pnts" and "x_rng", where x is the name given to the variables
    independent variables in data vault
    """
    dv.add_parameter('n0_rng', cfg['meas_parameters']['n0_rng'])
    dv.add_parameter('p0_pnts', cfg['meas_parameters']['p0_pnts'])
    dv.add_parameter('n0_pnts', cfg['meas_parameters']['n0_pnts'])
    dv.add_parameter('p0_rng', cfg['meas_parameters']['p0_rng'])

    dv.add_parameter('measurement_type','transport')


    if kwargs is not None:
        for key, value in kwargs.items():
            dv.add_parameter(key, value)

def main():
    # Loads config
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    lockins = cfg['measurement']['lockins']
    gates = cfg['measurement']['gates']

    measurement = cfg['measurement']

    lockin1_settings = cfg['lockin1']
    if lockins == 2:
        lockin2_settings = cfg['lockin2']
    meas_parameters = cfg['meas_parameters']
    delta_var = meas_parameters['delta_var']

    voltage_mult = cfg['measurement']['multiplier']

    # Connections and Instrument Configurations
    cxn = labrad.connect()
    dv = cxn.data_vault
    create_file(dv, cfg)

    # setting gate sweep settings, if vt and vb are flipped, it changes which is 'X' and 'Y' to match the output of function_select
    if gates==1:
        gate_ch1 = cfg['gate1']['ch']
        X_MIN = cfg['gate1']['limits'][0]
        X_MAX = cfg['gate1']['limits'][1]
        Y_MIN = -10.
        Y_MAX = 10.
    elif gates==2:
        if cfg['gate1']['type'] == 'vt':
            gate_ch1 = cfg['gate1']['ch']
            X_MIN = cfg['gate1']['limits'][0]
            X_MAX = cfg['gate1']['limits'][1]
            gate_ch2 = cfg['gate2']['ch']
            Y_MIN = cfg['gate2']['limits'][0]
            Y_MAX = cfg['gate2']['limits'][1]
        elif cfg['gate1']['type'] == 'vb':
            gate_ch2 = cfg['gate1']['ch']
            Y_MIN = cfg['gate1']['limits'][0]
            Y_MAX = cfg['gate1']['limits'][1]
            gate_ch1 = cfg['gate2']['ch']
            X_MIN = cfg['gate2']['limits'][0]
            X_MAX = cfg['gate2']['limits'][1]

    t0 = time.time()

    pxsize = (meas_parameters['n0_pnts'], meas_parameters['p0_pnts'])
    extent = (meas_parameters['n0_rng'][0], meas_parameters['n0_rng'][1], meas_parameters['p0_rng'][0], meas_parameters['p0_rng'][1])
    num_x = pxsize[0]
    num_y = pxsize[1]
    print extent, pxsize


    DELAY_MEAS = 3.0 * cfg['lockin1']['tc'] * 1e6
    SWEEP_MULT = 30.0 #how many actual points are taken for each DELAY_MEAS

    est_time = (pxsize[0] * pxsize[1] + pxsize[1]) * DELAY_MEAS * 1e-6 / 60.0
    dt = pxsize[0]*DELAY_MEAS*1e-6/60.0
    print("Will take a total of {} mins. With each line trace taking {} ".format(est_time, dt))

    m, mdn = mesh(0.0, offset=(0, -0.0), drange=(extent[2], extent[3]),
                  nrange=(extent[0], extent[1]), gates=cfg['measurement']['gates'],
                  pxsize=pxsize, delta=delta_var)

    for i in range(num_y):

        Ix = np.zeros(num_x)
        Iy = np.zeros(num_x)
        Vx = np.zeros(num_x)
        Vy = np.zeros(num_x)

        vec_x = m[i, :][:, 0]/voltage_mult
        vec_y = m[i, :][:, 1]/voltage_mult

        md = mdn[i, :][:, 0]
        mn = mdn[i, :][:, 1]

        mask = np.logical_and(np.logical_and(vec_x <= X_MAX/voltage_mult, vec_x >= X_MIN/voltage_mult),
                              np.logical_and(vec_y <= Y_MAX/voltage_mult, vec_y >= Y_MIN/voltage_mult))

        if np.any(mask == True):
            start, stop = np.where(mask == True)[0][0], np.where(mask == True)[0][-1]

            if gates==1:
                vstart = [vec_x[start]]
                vstop = [vec_x[stop]]
            if gates==2:
                vstart = [vec_x[start], vec_y[start]]
                vstop = [vec_x[stop], vec_y[stop]]

#
            num_points = stop - start + 1
            print("{} of {}  --> Ramping. Points: {}".format(i + 1, num_y, num_points))


            Vx = np.sin(vec_x+vec_y)
            Vy = np.sin(vec_x-vec_y)
            Ix = np.cos(vec_x*10.)
            Ix = np.cos(vec_y*10.)


        R = Vx/Ix
        sig = Ix/Vx
        j = np.linspace(0, num_x - 1, num_x)
        ii = np.ones(num_x) * i
        t1 = np.ones(num_x) * time.time() - t0

        totdata = np.array([j, ii, vec_x, vec_y, Ix, Iy, Vx, Vy, md, mn, R, sig, t1])
        time.sleep(1        )
        dv.add(totdata.T)

    print("it took {} s. to write data".format(time.time() - t0))


if __name__ == '__main__':
    main()
