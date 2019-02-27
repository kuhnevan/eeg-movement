# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# %matplotlib qt5

from scipy.io import loadmat
import mne

data_file_path = "../../data/s01.mat";

mat = loadmat(data_file_path)

eeg = mat["eeg"]["movement_left"] # mat["data"][0, 3]["X"][0, 0] * 10e-6

ch_names = [" 1 ",
" 2 ",
" 3 ",
" 4 ",
" 5 ",
" 6 ",
" 7 ",
" 8 ",
" 9 ",
" 10 ",
" 11 ",
" 12 ",
" 13 ",
" 14 ",
" 15 ",
" 16 ",
" 17 ",
" 18 ",
" 19 ",
" 20 ",
" 21 ",
" 22 ",
" 23 ",
" 24 ",
" 25 ",
" 26 ",
" 27 ",
" 28 ",
" 29 ",
" 30 ",
" 31 ",
" 32 ",
" 33 ",
" 34 ",
" 35 ",
" 36 ",
" 37 ",
" 38 ",
" 39 ",
" 40 ",
" 41 ",
" 42 ",
" 43 ",
" 44 ",
" 45 ",
" 46 ",
" 47 ",
" 48 ",
" 49 ",
" 50 ",
" 51 ",
" 52 ",
" 53 ",
" 54 ",
" 55 ",
" 56 ",
" 57 ",
" 58 ",
" 59 ",
" 60 ",
" 61 ",
" 62 ",
" 63 ",
" 64 ", "EMG1", "EMG2", "EMG3", "EMG4"]

info = mne.create_info(ch_names, 512, ch_types=["eeg"] * 64 + ["emg"] * 4)
raw = mne.io.RawArray(eeg[0][0], info)

# raw.set_montage("biosemi64")

raw_tmp = raw.copy()
raw_tmp.filter(1, None)

ica = mne.preprocessing.ICA(method="extended-infomax", random_state=1)
ica.fit(raw_tmp)

# ica.plot_components(inst=raw_tmp)

# ica.plot_sources(inst=raw_tmp)

ica.exclude = [1]

raw_corrected = raw.copy()
ica.apply(raw_corrected)

raw.plot(n_channels=64, start=54, duration=4, 
         scalings=dict(eeg=512, emg=512))

raw_corrected.plot(n_channels=64, start=54, duration=4, 
                   scalings=dict(eeg=512, emg=512))
