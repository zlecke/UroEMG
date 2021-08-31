#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import uro_utilities
from uro_utilities import parse_cmg, find_void_times

from uro_utilities.wavelets import calculate_wavelets, calculate_wavelet_spectrum


def void_time_wavelet(study_path, participant_id=None, progress=None):
    if progress:
        fraction_complete = 0
        progress((fraction_complete, 'Finding Data Files'))

    if participant_id is None:
        participant_id = int(os.path.basename(os.path.dirname(study_path))[:3])

    study_letter = os.path.basename(study_path)[
                   os.path.basename(study_path).find(str(participant_id)) + 3:
                   os.path.basename(study_path).find(str(participant_id)) + 7]

    void_p_n = {study_letter: {}}
    center_freqs = {study_letter: {}}
    time_res = {study_letter: {}}
    bandwidths = {study_letter: {}}

    f_files = {'uro': {}, 'lab': {}, 'comments': {}}

    for root, dirs, files in os.walk(study_path):
        for f in files:
            if f.endswith('.txt') \
                    and 'Data Acquisition' in root \
                    and 'Laborie' in root \
                    and study_letter in f:
                f_files['lab'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

            if (f.endswith('data.txt')
                    and ('Tracking Files' in root or 'Tracking Notes' in root)
                    and 'Data Acquisition' in root):
                f_files['uro'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

            if f.endswith('_comments.txt') and ('pre-processing' == os.path.basename(root).lower()
                                                or
                                                'pre processing' == os.path.basename(root).lower()):
                f_files['comments'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

    for i, cmg in enumerate(sorted(f_files['uro'].keys())):
        if cmg not in f_files['lab'].keys():
            continue

        uro_name = f_files['uro'][cmg]
        lab_name = f_files['lab'][cmg]
        com_name = f_files['comments'].get(cmg, '')

        if progress:
            fraction_complete = (1./4.) * ((i + 1)/len(f_files['uro'].keys()))
            progress((fraction_complete, "Reading Data - {}".format(cmg)))

        lab_df = uro_utilities.read_uro_laborie(lab_name,
                                                columns=['EMGR2_A'],
                                                round=4).set_index('Time')

        void_times = find_void_times(com_name)

        uro_df = pd.read_csv(uro_name, delimiter='\t')

        if progress:
            fraction_complete = (1./2.) * ((i + 1)/len(f_files['uro'].keys()))
            progress((fraction_complete, "Finding Void Times - {}".format(cmg)))

        cough_time = uro_utilities.get_cough_time(uro_name)

        if len(void_times) == 0:

            void_1_time = np.nan if not uro_df.query('`Void Attemp 1` > 0').shape[0] > 0\
                else int(
                    np.rint(uro_df.query('`Void Attemp 1` > 0').loc[:, 'Time(Sec)'].iloc[
                                0])) - cough_time

            void_2_time = np.nan if not uro_df.query('`Void Attemp 2` > 0').shape[0] > 0\
                else int(
                    np.rint(uro_df.query('`Void Attemp 2` > 0').loc[:, 'Time(Sec)'].iloc[
                                0])) - cough_time

            void_3_time = np.nan if not uro_df.query('`Void Attemp 3` > 0').shape[0] > 0\
                else int(
                    np.rint(uro_df.query('`Void Attemp 3` > 0').loc[:, 'Time(Sec)'].iloc[
                                0])) - cough_time

            void_times = []
            if not np.isnan(void_1_time):
                void_times.append(void_1_time)
            if not np.isnan(void_2_time):
                void_times.append(void_2_time)
            if not np.isnan(void_3_time):
                void_times.append(void_3_time)

        try:
            void_emg = [lab_df.query('Time > @vtime - 5 and Time < @vtime + 5').loc[:, 'EMGR2_A']
                        for vtime in void_times]
        except KeyError:
            void_emg = [lab_df.query('Time > @vtime - 5 and Time < @vtime + 5').loc[:, 'EMG2_A']
                        for vtime in void_times]
        void_emg = [x for x in void_emg if x.shape[0] > 0]
        void_p_n[study_letter][cmg] = []
        center_freqs[study_letter][cmg] = []
        time_res[study_letter][cmg] = []
        bandwidths[study_letter][cmg] = []
        for j, vemg in enumerate(void_emg):
            if progress:
                fraction_complete = (1./2.) * ((i + 1)/len(f_files['uro'].keys())) \
                                    + (1./4.) * ((j + 1)/len(void_emg))
                progress((fraction_complete,
                          "Analyzing EMG - {} - Void Attempt {}".format(cmg, j + 1)))
            F_psi, cf, tres, bandwidth = calculate_wavelets(vemg.shape[0],
                                                            sampling_rate=1 / lab_df.index.values[
                                                                1])
            void_p_n[study_letter][cmg].append(pd.DataFrame(data=calculate_wavelet_spectrum(
                    vemg.values.ravel(),
                    F_psi,
                    len(F_psi[0, :]),
                    cf,
                    tres),
                    index=vemg.index.to_series().apply(pd.to_timedelta, unit='sec'),
                    dtype=np.float64))
            center_freqs[study_letter][cmg].append(cf)
            time_res[study_letter][cmg].append(tres)
            bandwidths[study_letter][cmg].append(bandwidth)

    return void_p_n, center_freqs, time_res, bandwidths
