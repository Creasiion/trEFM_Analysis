#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 14:29:25 2023

@author: imani
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as stComp
import ffta
import mpld3
from ffta.simulation.mechanical_drive import MechanicalDrive
from ffta.simulation.utils import excitation

plt.rc('font', size=16)


def defaults():
    '''
    Sets the default value for cantilever dictionaries of Simulation and Calibration Curve pages.

    '''
        st.session_state['cantilever_dict'] = {
            'amp_invols':1.15821e-07,
            'def_invols':1.15821e-07,
            'soft_amp':0.3,
            'drive_freq':279173,
            'res_freq':279173,
            'k':20,
            'q_factor':450
            }

        st.session_state['force_dict'] = {
            'es_force':3.72358e-08,
            'delta_freq':-170, 
            'tau':50e-6
            }
        st.session_state['simulation_dict'] = {
            'trigger':0.0004,
            'total_time':0.002,
            'sampling_rate':1e7
            }
        st.session_state['others'] = {
            'beta':0.4,
            'tau_stretched':50e-6,
            'exponential_type':'Single Exponential',
            'tau1':50e-6,
            'tau2':500e-6
            }
        st.session_state['CC_cantilever_dict'] = {
            'amp_invols':1.15821e-07,
            'def_invols':1.15821e-07,
            'soft_amp':0.3,
            'drive_freq':279173,
            'res_freq':279173,
            'k':20,
            'q_factor':450
            }

        st.session_state['CC_force_dict'] = {
            'es_force':3.72358e-08,
            'delta_freq':-170, 
            'tau':50e-6
            }
        st.session_state['CC_simulation_dict'] = {
            'trigger':0.0004,
            'total_time':0.002,
            'sampling_rate':1e7
            }
        st.session_state['CC_others'] = {
            'beta':0.4,
            'tau_stretched':50e-6,
            'exponential_type':'Single Exponential',
            'tau1':50e-6,
            'tau2':500e-6
            }
        st.session_state['Tfp_Data_Analysis'] = None

def update_dict(dict_name, key):
    '''
    Updates part of dictionary when user inputs their value
    
    Parameters
    ----------
    dict_name : dict
        A dictionary that is within Streamlit's session state.
    key : string
        A key name apart of the dictionary dict_name.
    
    '''
    st.session_state[dict_name][key] = st.session_state[key]

def copy_values(ourDict, dictUsed):
    '''
    Parameters
    ----------
    ourDict : dict
        Dictionary in session state where values will be copied to.
    dictUsed : dict
        Dictionary where values were copied from.

    '''
    target_dict = st.session_state[ourDict]
    source_dict = st.session_state[dictUsed]
    for key in target_dict:
        if key in source_dict:
            target_dict[key] = source_dict[key]
        else:
            st.write('ERROR: Dictionary values can not be copied.')
            
def copy_ALL_values(dict1A, dict2A, dict1B, dict2B, dict1C, dict2C):
    copy_values(dict1A, dict2A)      
    copy_values(dict1B, dict2B)      
    copy_values(dict1C, dict2C)           
    

def plot_deflection(Cantilever, Z):
    '''
    Plots the deflection of a simulated cantilever

    Parameters
    ----------
    Cantilever : ffta.simulation.mechanical_drive.MechanicalDrive
        FFTA cantilever object.
    Z : ndarray
        Deflection from cantilever.

    '''
    fig = plt.figure()
    plt.plot(Cantilever.t_Z * 1e3, Z)
    plt.xlabel('Time (ms)')
    plt.ylabel('Deflection (V)',rotation='horizontal', loc='top', labelpad=-57)
    plt.title('Simulation with time constant = ' + str(Cantilever.tau) + ' s')
    fig_html = mpld3.fig_to_html(fig)
    stComp.html(fig_html, height=600)
    
def get_pix(Cantilever, Z, genre, sim_parms, region_oi):
    return ffta.pixel.Pixel(Z/Cantilever.amp_invols, method=genre, trigger=sim_parms['trigger'], total_time=sim_parms['total_time'], roi = region_oi)

def plot_pix(pixel):
    '''

    Parameters
    ----------
    pixel : ffta.pixel.Pixel
        FFTA point scan object.

    '''
    pixel.analyze()
    fig = pixel.plot()
    st.pyplot(fig)
 
    
def crop_data(time, data, time_start, time_stop):
    '''
    
    Parameters
    ----------
    time : ndarray
        Original time from data 
    data : ndarray
        Data that holds instantaneous frequency 
    time_start : float
        Time for simulation pix to use (ms)
    time_stop : float
        Time for simulation pix to use (ms)

    Returns
    -------
    crop_time : ndarray
        Shortens the time given to us based on time start and stop
    spliced_data : ndarray
        Crops the data given to us based on time start and stop

    '''
    start_ind = np.where(time>time_start)[0][0]
    stop_ind = np.where(time>time_stop)[0][0]
    spliced_data = data[start_ind:stop_ind]
    crop_time = time[start_ind:stop_ind]
    return crop_time, spliced_data

def normalize_data(data):
    norm_data = np.interp(data, (data.min(), data.max()), (0,1))
    return norm_data

 
def plot_inst_freq(time, data, y_label):
    '''
    Plots the instantaneous frequency

    Parameters
    ----------
    time : ndarray
        x-axis points
    data : ndarray
        y-axis points
    y_label : string
        Other part of y-label axis title

    '''
    fig = plt.figure()
    plt.plot(time, data)
    plt.xlabel('Time (ms)')
    plt.ylabel(y_label +'(Hz)')
    st.pyplot(fig)