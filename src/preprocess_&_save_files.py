# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:29:06 2019

@author: Kevin
"""
from scipy.io import loadmat
from scipy.io import savemat
import mne
import numpy as np
#import glob
#for filepath in glob.iglob("../../data/*.mat"):
    #print(filepath)
    # -*- coding: utf-8 -*-
# %matplotlib qt5
for j in range(1, 53):
    data_file_path = ""
    if j >= 10:
        data_file_path = "../../data/s{}.mat".format(j)
    else:
        data_file_path = "../../data/s0{}.mat".format(j)
    
    mat = loadmat(data_file_path)
    data_FFT_left = 0
    data_FFT_right = 0
    hand = ""
    ch_names = [" FP1 ", " AF7 ", " AF3 ", " F1 ", " F3 ", " F5 ", " F7 ", " FT7 ",
    " FC5 ", " FC3 ", " FC1 ", " C1 ", " C3 ", " C5 ", " T7 ", " TP7 ", " CP5 ",
    " CP3 ", " CP1 ", " P1 ", " P3 ", " P5 ", " P7 ", " P9 ", " PO7 ", " PO3 ",
    " O1 ", " Iz ", " Oz ", " POz ", " Pz ", " CPZ ", " FPZ ", " FP2 ", " AF8 ",
    " AF4 ", " AFZ ", " FZ ", " F2 ", " F4 ", " F6 ", " F8 ", " FT8 ", " FC6 ",
    " FC4 ", " FC2 ", " FCz ", " Cz ", " C2 ", " C4 ", " C6 ", " T8 ", " TP8 ",
    " CP6 ", " CP4 ", " CP2 ", " P2 ", " P4 ", " P6 ", " P8 ", " P10 ", " PO8 ",
    " PO4 ", " O2 ", "EMG1", "EMG2", "EMG3", "EMG4"]
    #Create new info and raw array for movement event information
    ch_names_events = ch_names + ["movement_event"]
    
    #preprocessing for both hands of the subject
    for i in range(2):
        if i == 0:
            hand = "movement_left"
        else:
            hand = "movement_right"
        eeg = mat["eeg"][hand]
        movement_event = mat["eeg"]["movement_event"][0][0]
        eeg = np.concatenate((eeg[0][0], movement_event), axis=0)
        info = mne.create_info(ch_names_events, 512, ch_types=["eeg"] * 64 + ["emg"] * 4 + ["misc"] * 1)
        
        raw = mne.io.RawArray(eeg, info)
        
        events = mne.find_events(raw, stim_channel='movement_event')
        
        raw.set_montage("biosemi64")
        
        raw_tmp = raw.copy()
        raw_tmp.filter(1, None)
        
        ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
        ica.fit(raw_tmp)
        
        ica.plot_components(inst=raw_tmp)
        
        ica.plot_sources(inst=raw_tmp)
        
        ica.exclude = [0]
        
        raw_corrected = raw.copy()
        ica.apply(raw_corrected)
        
        raw.plot(n_channels=64, start=54, duration=4, 
            scalings=dict(eeg=512, emg=512))
            
        raw_corrected.plot(n_channels=64, start=54, duration=4, 
            scalings=dict(eeg=512, emg=512))
        
        #create epoched data
        tmin = -2  # In seconds
        tmax = 5
        epochs = mne.Epochs(raw, events, tmin=tmin, tmax=tmax)
        epochs_data = epochs.get_data()
        
#        if i==0:
#            data_FFT_left = np.fft.fft(epochs_data, axis=0)
#        else:
#            data_FFT_right = np.fft.fft(epochs_data, axis=0)
            
    #end inner for loop
    savemat('../../processed-data/s{}-left.mat'.format(j),
            {'movement_left': data_FFT_left})
    savemat('../../processed-data/s{}-right.mat'.format(j),
            {'movement_right': data_FFT_right})
#end outer for loop
########
