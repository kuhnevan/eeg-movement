# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %matplotlib qt5

from scipy.io import loadmat
import mne

data_file_path = "../../data/s01.mat"

mat = loadmat(data_file_path)

eeg = mat["eeg"]["movement_left"] # mat["data"][0, 3]["X"][0, 0] * 10e-6

ch_names = [" FP1 ",
" AF7 ",
" AF3 ",
" F1 ",
" F3 ",
" F5 ",
" F7 ",
" FT7 ",
" FC5 ",
" FC3 ",
" FC1 ",
" C1 ",
" C3 ",
" C5 ",
" T7 ",
" TP7 ",
" CP5 ",
" CP3 ",
" CP1 ",
" P1 ",
" P3 ",
" P5 ",
" P7 ",
" P9 ",
" PO7 ",
" PO3 ",
" O1 ",
" lz ",
" Oz ",
" POz ",
" Pz ",
" CPZ ",
" FPZ ",
" FP2 ",
" AF8 ",
" AF4 ",
" AFZ ",
" FZ ",
" F2 ",
" F4 ",
" F6 ",
" F8 ",
" FT8 ",
" FC6 ",
" FC4 ",
" FC2 ",
" FCz ",
" Cz ",
" C2 ",
" C4 ",
" C6 ",
" T8 ",
" TP8 ",
" CP6 ",
" CP4 ",
" CP2 ",
" P2 ",
" P4 ",
" P6 ",
" P8 ",
" P10 ",
" PO8 ",
" PO4 ",
" O2 ", "EMG1", "EMG2", "EMG3", "EMG4"]

info = mne.create_info(ch_names, 512, ch_types=["eeg"] * 64 + ["emg"] * 4)
raw = mne.io.RawArray(eeg[0][0], info)

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
