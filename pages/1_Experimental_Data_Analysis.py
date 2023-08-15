#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:40:01 2023

@author: imani
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as stComp
import mpld3
import ffta
from ffta.simulation.mechanical_drive import MechanicalDrive
from ffta.simulation.utils import excitation
from Functions import defaults, plot_deflection, crop_data, normalize_data, plot_inst_freq
import igor.binarywave as bw

plt.rc('font', size=16)

if 'cantilever_dict' not in st.session_state:
    defaults()

st.session_state['uploaded_file'] = st.file_uploader("Upload your file here...", label_visibility='collapsed')
uploaded_file = st.session_state['uploaded_file']

if st.session_state['uploaded_file'] is not None:
    st.write('Current File: ' + uploaded_file.name)
    
    #Extract necessary data from.ibw file
    file = bw.load(uploaded_file)
    signal = file['wave']['wData']
    average_signal = np.mean(signal,axis=1)
    
    time = np.linspace(0, 2, average_signal.shape[0])
    fig = plt.figure()
    plt.plot(time, average_signal)
    plt.xlabel('Time (ms)')
    plt.ylabel('Deflection (V)',rotation='horizontal', loc='top', labelpad=-57)
    fig_html = mpld3.fig_to_html(fig)
    stComp.html(fig_html, height=600)
    
    col1, col2, col3 = st.columns(3)
    with col1: #Method type and pix creation
        method_type = st.radio("Method type", ('STFT', 'Hilbert'))
        roi_ = st.number_input("Region of Interest (roi)", value = 0.0005, format='%f')
        pix = ffta.pixel.Pixel(average_signal, method=method_type.lower(),trigger=0.0004, total_time=0.002, roi=roi_)
        
    with col2: #N Taps
        n_taps_val = st.text_input('N taps', value=None)
        if n_taps_val == 'None' or n_taps_val == '':
            n_taps_val = None
        elif n_taps_val.isnumeric():
            n_taps_val = int(n_taps_val)
            pix.n_taps = n_taps_val
        else:
            st.error('Input must be an integer')
            
    with col3: #FFT Time Res
        time_res = st.text_input('FFT time res ($\mu s$)', value=80, help='Will convert microseconds to seconds')
        if time_res == 'None' or time_res == '':
            time_res = None
        elif time_res.isnumeric():
            time_res = int(time_res)
            pix.fft_time_res = time_res * 1e-6 #converting microseconds to seconds
        else:
            st.error('Input must be an integer')
            
    pix.analyze()
    
    #Save necessary data from uploaded file that can be used in a seperate page to session state
    st.session_state['InstFreq_Data_Analysis'] = pix.inst_freq
    st.session_state['Tfp_Data_Analysis'] = pix.tfp * 1e6

    user_start_time = st.number_input('Start time (ms)', value = 0.2, format='%f')
    user_end_time = st.number_input('End time (ms)', value = 1.8, format='%f')
    
    time, plotted_data = crop_data(time, pix.inst_freq, user_start_time, user_end_time)
   
    tab1, tab2 = st.tabs(['Raw Data', 'Normalized Data'])
    with tab1:
        tab1col1, tab1col2 = st.columns([0.84, 0.16])
        with tab1col1:
            plot_inst_freq(time, plotted_data, 'Inst. Frequency ')
        with tab1col2:
            st.write('$t_{FP}$: ' + str(pix.tfp*1e6) + '$\mu s$' )

        
    with tab2:
        tab2col1, tab2col2 = st.columns([0.84, 0.16])
        with tab2col1:
            norm_pix = normalize_data(plotted_data)
            plot_inst_freq(time, norm_pix, 'Normalized Frequency ') 
        with tab2col2:
            st.write('$t_{FP}$: ' + str(pix.tfp*1e6) + '$\mu s$' )
else:
    st.write('Current File: ')   
