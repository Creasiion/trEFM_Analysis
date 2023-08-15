#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:53:13 2023

@author: imani
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as stComp
import mpld3
import ffta
import pandas as pd
from ffta.simulation.mechanical_drive import MechanicalDrive
from ffta.simulation.utils import excitation
from Functions import defaults, update_dict, copy_ALL_values, crop_data, normalize_data, plot_inst_freq
from scipy import interpolate

plt.rc('font', size=16)

if 'cantilever_dict' not in st.session_state:
    defaults()
    
if 'desired_tfp' not in st.session_state:
    st.session_state['desired_tfp'] = 75
else:
    st.session_state['desired_tfp'] = st.session_state['desired_tfp']

def import_tfp():
    '''
    Gets tfp from session state that was taken in experimental data analysis page.
    
    '''
    if st.session_state['Tfp_Data_Analysis'] is None:
        st.write('No file has been uploaded')
    else:
        st.session_state['desired_tfp'] = st.session_state['Tfp_Data_Analysis']
    
def tfp_to_tau_calibration(cantilever, tau_values, tfp_output):
    '''
    Simulates multiple taus and saves tfps, both are used to create calibration curve.

    Parameters
    ----------
    cantilever : ffta.simulation.mechanical_drive.MechanicalDrive
        FFTA cantilever object.
    tau_values : array
        List of tau values to generate cantilvers from.
    tfp_output : array
        List of Tfp values correlated from tau values. Empty array.

    Returns
    -------
    tfp_output : array
        List of Tfp values correlated from tau values.

    '''
    for i in range(len(tau_values)):
        t = tau_values[i]
        t = t*1e-6 # converts from seconds to microseconds
        cantilever.func_args = [t] #need to use .func_args to change tau, and value should be in []
        X, info = cantilever.simulate()
        pix = ffta.pixel.Pixel(X/cantilever.amp_invols, method='stft',trigger=st.session_state['CC_simulation_dict']['trigger'], total_time=st.session_state['CC_simulation_dict']['total_time'], roi=1e-3)
        pix.fft_time_res = 60e-6
        pix.analyze()
        tfp_output.append(pix.tfp*1e6) # converts from seconds to microeconds
    return tfp_output

def plot_calibration(tau_list, tfp_list):
    fig = plt.figure()
    plt.scatter(tau_list,tfp_list)
    plt.plot(tau_list,tfp_list)
    plt.xlabel('$\\tau$ ($\mu s$)')
    plt.ylabel('$t_{FP}$ ($\mu s$)')
    st.pyplot(fig)
    
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
    
cant_parms = st.session_state['CC_cantilever_dict']
force_parms = st.session_state['CC_force_dict']
sim_parms = st.session_state['CC_simulation_dict']
other_parms = st.session_state['CC_others']

for key in cant_parms:
    st.session_state[key] = cant_parms[key]
for key in force_parms:
    st.session_state[key] = force_parms[key]
for key in sim_parms:
    st.session_state[key] = sim_parms[key]
for key in other_parms:
    st.session_state[key] = other_parms[key]


st.button('Copy from Simulation', on_click=copy_ALL_values, args=['CC_cantilever_dict','cantilever_dict','CC_force_dict','force_dict','CC_simulation_dict','simulation_dict'])   

with st.expander('Cantilever Parameters', expanded = False):
    amp_invols = st.number_input('Amplitude (m/V)', key='amp_invols', on_change=update_dict, args=['CC_cantilever_dict', 'amp_invols'], format='%e')
    def_invols = st.number_input('Deflection (m/V)', key='def_invols', on_change=update_dict, args=['CC_cantilever_dict', 'def_invols'], format='%e')
    soft_amp = st.number_input('Soft Amplitude (m/V)', key='soft_amp', on_change=update_dict, args=['CC_cantilever_dict', 'soft_amp'], format='%f')
    drive_freq = st.number_input('Driving Frequency (Hz)', key='drive_freq', on_change=update_dict, args=['CC_cantilever_dict', 'drive_freq'])
    res_freq = st.number_input('Resonance Frequency (Hz)', key='res_freq', on_change=update_dict, args=['CC_cantilever_dict', 'res_freq'])
    k = st.number_input('Spring Constant (N/m)', key='k', on_change=update_dict, args=['CC_cantilever_dict', 'k'])
    q_factor = st.number_input('Cantilever Quality', key='q_factor', on_change=update_dict, args=['CC_cantilever_dict', 'q_factor'])
    
with st.expander('Force Parameters', expanded = False):
    es_force = st.number_input('Electrostatic Force (N)', key='es_force', on_change=update_dict, args=['CC_force_dict', 'es_force'], format='%e')
    delta_freq = st.number_input('Delta Frequency (Hz)', key='delta_freq', on_change=update_dict, args=['CC_force_dict', 'delta_freq'])
    tau = st.number_input('Tau (s)', key='tau', on_change=update_dict, args=['CC_force_dict', 'tau'], format='%e')
    
with st.expander('Simulation Parameters', expanded = False):
    trigger = st.number_input('Trigger (s)', key='trigger', on_change=update_dict, args=['CC_simulation_dict', 'trigger'], format='%f')
    total_time = st.number_input('Total Time(s)', key='total_time', on_change=update_dict, args=['CC_simulation_dict', 'total_time'], format='%f')
    sampling_rate = st.number_input('Sampling Rate (Hz)', key='sampling_rate', on_change=update_dict, args=['CC_simulation_dict', 'sampling_rate'], format='%e') 

load_CC = st.button('Form Calibration Curve')
if 'load_state_CC' not in st.session_state:
    st.session_state.load_state_CC = False
        
if load_CC or st.session_state.load_state_CC:
    st.session_state.load_state_CC = True
  
    cant = MechanicalDrive(cant_parms, force_parms, sim_parms)
    Z, info = cant.simulate()
    
    
    col1, col2 = st.columns([0.16, 0.84])
    
    with col1: #User input tau values to form calibration curve
        min_tau = st.number_input('Minumum Tau value', value=1)
        max_tau = st.number_input('Maximum Tau value', value=100)
        num_taus = st.number_input('Amount of taus', value=15)
        taus = np.linspace(min_tau, max_tau, num_taus)
        tfps = []
        tfp_to_tau_calibration(cant, taus, tfps)
    with col2:
        plot_calibration(taus, tfps)
        
    calibration_df = pd.DataFrame({'taus us':taus,'tfps us':tfps})
    csv = convert_df(calibration_df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='Calibration_Curve.csv',
        mime='text/csv',
    )
    
    #Estimate tau using interpolate
    st.caption('Estimate tau')
    
    imported_tfp = st.button('Import tfp from Data Analysis') #Import tau from Experimental Data Analysis page
    if imported_tfp:
        import_tfp()
        
    desired_tfp = st.number_input('$t_{FP}$ ($\mu s$)', key='desired_tfp', format='%f')

    guesstimate = interpolate.interp1d(tfps, taus)
    estimated_tau = float(guesstimate(desired_tfp))
    st.write('Estimated Calibrated Tau: ' + str(estimated_tau) + ' ($\mu s$)')
    
    st.divider()
    
    #Plot the user data (instantaneous frequency) on the same axis as a simulated cantilever with the same tau value (from the interpolate function) 
    st.header('Plot User Instantaneous Frequency with Simulation (using estimated tau)')
    
    trigger_ms =  sim_parms['trigger'] *1e3
    total_time_ms = sim_parms['total_time'] * 1e3
    default_start_time = 0.5 * trigger_ms
    default_end_time = 0.9 * total_time_ms
    
    user_start_time = st.number_input('Start time (ms)', value = default_start_time, format='%f')
    user_end_time = st.number_input('End time (ms)', value = default_end_time, format='%f')
    
    
    sim_and_user = st.button('Plot Together')
    if 'sim_and_user_CC' not in st.session_state:
        st.session_state.sim_and_user_CC = False
        
    if sim_and_user or st.session_state.sim_and_user_CC:
        st.session_state.sim_and_user_CC = True
        
        cant_1 = MechanicalDrive(cant_parms, force_parms, sim_parms, func=excitation.single_exp, func_args=[estimated_tau*1e-6]) #Creating cantilever with estimated tau
        Z, info = cant_1.simulate()
        time_ms = cant_1.t_Z * 1e3
        sim_pix = ffta.pixel.Pixel(Z/cant_1.amp_invols, method='stft',trigger=0.0004, total_time=0.004, roi=0.0005)
        sim_pix.fft_time_res = 120e-6
        sim_pix.analyze()
        user_freq = st.session_state['InstFreq_Data_Analysis']
        
        time, sim_data = crop_data(time_ms, sim_pix.inst_freq, user_start_time, user_end_time)
        time, user_data = crop_data(time_ms, user_freq, user_start_time, user_end_time)
        
        norm_sim_data = normalize_data(sim_data)
        norm_user_data = normalize_data(user_data)
        
        fig = plt.figure()
        plt.plot(time, norm_sim_data, label='Simulation Estimate')
        plt.plot(time, norm_user_data, label='User Data')
        plt.xlabel('Time (ms)')
        plt.ylabel('Normalized Instantaneous Frequencies (Hz)')
        plt.legend()
        st.pyplot(fig)
    
    