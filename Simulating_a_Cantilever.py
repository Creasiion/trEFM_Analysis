#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as stComp
import mpld3
import ffta
from ffta.simulation.mechanical_drive import MechanicalDrive
from ffta.simulation.utils import excitation
from Functions import defaults, plot_deflection, get_pix,crop_data, normalize_data, plot_inst_freq, update_dict

plt.rc('font', size=16)

if 'cantilever_dict' not in st.session_state:
    defaults()
 
    
#Choose parameters for cantilever simulation

with st.sidebar:
    cant_parms = st.session_state['cantilever_dict']
    force_parms = st.session_state['force_dict']
    sim_parms = st.session_state['simulation_dict']
    other_parms = st.session_state['others']

    for key in cant_parms:
        st.session_state[key] = cant_parms[key]
    for key in force_parms:
        st.session_state[key] = force_parms[key]
    for key in sim_parms:
        st.session_state[key] = sim_parms[key]
    for key in other_parms:
        st.session_state[key] = other_parms[key]
        
    exponential_type = st.selectbox('', ('Single Exponential','Stretched Exponential', 'Bi Exponential'), key='exponential_type', on_change=update_dict, args=['others', 'exponential_type'])
    
    if other_parms['exponential_type'] == 'Stretched Exponential':
        beta = st.number_input('Beta', key='beta',on_change=update_dict, args=['others', 'beta'], format='%f')
        tau_stretched = st.number_input('Tau (s)', key='tau_stretched',on_change=update_dict, args=['others', 'tau_stretched'], format='%e')
        
    if other_parms['exponential_type'] == 'Bi Exponential':
        tau1 = st.number_input('Tau 1 (s)', key='tau1', on_change=update_dict, args=['others', 'tau1'], format='%e')
        tau2 = st.number_input('Tau 2 (s)', key='tau2',on_change=update_dict, args=['others', 'tau2'], format='%e')
        
    with st.expander('Cantilever Parameters', expanded = False):
        amp_invols = st.number_input('Amplitude (m/V)', key='amp_invols', on_change=update_dict, args=['cantilever_dict', 'amp_invols'], format='%e')
        def_invols = st.number_input('Deflection (m/V)', key='def_invols', on_change=update_dict, args=['cantilever_dict', 'def_invols'], format='%e')
        soft_amp = st.number_input('Soft Amplitude (m/V)', key='soft_amp', on_change=update_dict, args=['cantilever_dict', 'soft_amp'], format='%f')
        drive_freq = st.number_input('Driving Frequency (Hz)', key='drive_freq', on_change=update_dict, args=['cantilever_dict', 'drive_freq'])
        res_freq = st.number_input('Resonance Frequency (Hz)', key='res_freq', on_change=update_dict, args=['cantilever_dict', 'res_freq'])
        k = st.number_input('Spring Constant (N/m)', key='k', on_change=update_dict, args=['cantilever_dict', 'k'])
        q_factor = st.number_input('Cantilever Quality', key='q_factor', on_change=update_dict, args=['cantilever_dict', 'q_factor'])
        
    with st.expander('Force Parameters', expanded = False):
        es_force = st.number_input('Electrostatic Force (N)', key='es_force', on_change=update_dict, args=['force_dict', 'es_force'], format='%e')
        delta_freq = st.number_input('Delta Frequency (Hz)', key='delta_freq', on_change=update_dict, args=['force_dict', 'delta_freq'])
        if other_parms['exponential_type'] == 'Single Exponential':
            tau = st.number_input('Tau (s)', key='tau', on_change=update_dict, args=['force_dict', 'tau'], format='%e')
        else:
            tau = 50e-6
            
    with st.expander('Simulation Parameters', expanded = False):
        trigger = st.number_input('Trigger (s)', key='trigger', on_change=update_dict, args=['simulation_dict', 'trigger'], format='%f')
        total_time = st.number_input('Total Time (s)', key='total_time', on_change=update_dict, args=['simulation_dict', 'total_time'], format='%f')
        sampling_rate = st.number_input('Sampling Rate (Hz)', key='sampling_rate', on_change=update_dict, args=['simulation_dict', 'sampling_rate'], format='%e')   
     
        
    #load initial graphs that uses parameters given BEFORE pressing the button
    load = st.button('Load Graphs')
    if 'load_state' not in st.session_state:
        st.session_state.load_state = False
#End of sidebar
        
if load or st.session_state.load_state:
    st.session_state.load_state = True
    
    if other_parms['exponential_type'] == 'Stretched Exponential':
        cant = MechanicalDrive(cant_parms, force_parms, sim_parms, func=excitation.str_exp, func_args=[tau_stretched,beta])
    elif other_parms['exponential_type'] == 'Bi Exponential':
        cant = MechanicalDrive(cant_parms, force_parms, sim_parms, func=excitation.bi_exp, func_args=[tau1,tau2])
    else:
        cant = MechanicalDrive(cant_parms, force_parms, sim_parms)
        
    Z, info = cant.simulate()
    
    plot_deflection(cant, Z/cant.amp_invols)
    
    #Allowing user to input method type, n_taps, and fft_time_res for their data
    col1, col2, col3 = st.columns(3)
    with col1:
        method = st.radio("Method type", ('STFT', 'Hilbert'))
        roi = st.number_input("Region of Interest (roi)", value = 0.0005, format='%f')
        pix = get_pix(cant, Z, method.lower(), sim_parms, roi)
    with col2:
        n_taps_val = st.text_input('N taps', value=None)
        if n_taps_val == 'None' or n_taps_val == '':
            n_taps_val = None
        elif n_taps_val.isnumeric():
            n_taps_val = int(n_taps_val)
            pix.n_taps = n_taps_val
        else:
            st.error('Input must be an integer')
    with col3:
        time_res = st.text_input('FFT time res ($\mu s$)', value=80, help = 'Will convert microseconds to seconds')
        if time_res == 'None' or time_res == '':
            time_res = None
        elif time_res.isnumeric():
            time_res = int(time_res)
            pix.fft_time_res = time_res * 1e-6 #converting microseconds to seconds
        else:
            st.error('Input must be an integer')
            
    pix.analyze()

    time_ms = cant.t_Z * 1e3   
    trigger_ms =  sim_parms['trigger'] *1e3
    total_time_ms = sim_parms['total_time'] * 1e3
    default_start_time = 0.5 * trigger_ms
    default_end_time = 0.9 * total_time_ms
    
    user_start_time = st.number_input('Start time (ms)', value = default_start_time, format='%f')
    user_end_time = st.number_input('End time (ms)', value = default_end_time, format='%f')
    
    
    time, plotted_data = crop_data(time_ms, pix.inst_freq, user_start_time, user_end_time) #obtaining the data, plotted_data having the pix the user wants
    
    
    tab1, tab2 = st.tabs(['Raw Data', 'Normalized Data'])
    
    with tab1:
        tab1col1, tab1col2 = st.columns([0.84, 0.16])
        with tab1col1:
            plot_inst_freq(time, plotted_data, 'Frequency ')
        with tab1col2:
            if exponential_type == 'Single Exponential':
                st.write('$\\tau$: ' + str(force_parms['tau']*1e6) + '$\mu s$' )
            if exponential_type == 'Stretched Exponential':
                st.write('$\\tau$: ' + str(tau_stretched*1e6) + '$\mu s$' )
            if exponential_type == 'Bi Exponential':
                st.write('$\\tau$ 1: ' + str(tau1*1e6) + '$\mu s$' )
                st.write('$\\tau$ 2: ' + str(tau2*1e6) + '$\mu s$' )

            st.write('$t_{FP}$: ' + str(pix.tfp*1e6) + '$\mu s$' )
        
    with tab2:
        tab2col1, tab2col2 = st.columns([0.84, 0.16])
        with tab2col1:
            norm_pix = normalize_data(plotted_data)
            plot_inst_freq(time, norm_pix, 'Normalized Frequency ') 
        with tab2col2:
            st.write('$\\tau$: ' + str(force_parms['tau']*1e6) + '$\mu s$' )
            st.write('$t_{FP}$: ' + str(pix.tfp*1e6) + '$\mu s$' )

