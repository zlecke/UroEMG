#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import uro_utilities
from uro_utilities import parse_cmg, find_void_times

from uro_utilities.wavelets import calculate_wavelets, calculate_wavelet_spectrum


def void_time_wavelet(study_path, participant_id=None, progress=None):
    if participant_id is None:
        participant_id = int(os.path.basename(os.path.dirname(study_path))[:3])

    study_letter = os.path.basename(study_path)[
                   os.path.basename(study_path).find(str(participant_id)) + 3:
                   os.path.basename(study_path).find(str(participant_id)) + 7]

    void_p_n = {study_letter: {}}

    f_files = {'uro': {}, 'lab': {}}
    found_files = {'uro': [], 'lab': []}

    for root, dirs, files in os.walk(study_path):
        for f in files:
            if f.endswith('.txt') \
                    and 'Data Acquisition' in root \
                    and 'Laborie' in root \
                    and study_letter in f:
                f_files['lab'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

            if f.endswith('data.txt') and 'Tracking Files' in root and 'Data Acquisition' in root:
                f_files['uro'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

    for f in found_files['uro']:
        if f.upper().find('CMG') > -1:
            ci = f.upper().find('CMG')
            if f[ci + 3:f[ci:].upper().find('_') + ci].strip().isdigit():
                cmg = 'CMG{:d}'.format(int(f[ci + 3:f[ci:].upper().find('_') + ci].strip()))
            else:
                cmg = 'CMG{}'.format(
                    roman.fromRoman(f[ci + 3:f[ci:].upper().find('_') + ci].strip()))
        else:
            cmg = 'CMG1'
        f_files['uro'][cmg] = f

    for f in found_files['lab']:
        if f.upper().find('CMG') > -1:
            ci = f.upper().find('CMG')
            if f[ci + 3:f[ci:].upper().find('.') + ci].strip().isdigit():
                cmg = 'CMG{:d}'.format(int(f[ci + 3:f[ci:].upper().find('.') + ci].strip()))
            else:
                cmg = 'CMG{}'.format(
                        roman.fromRoman(f[ci + 3:f[ci:].upper().find('.') + ci].strip()))
        else:
            cmg = 'CMG1'
        f_files['lab'][cmg] = f

    for cmg in sorted(f_files['uro'].keys()):
        if cmg not in f_files['lab'].keys():
            continue

        uro_name = f_files['uro'][cmg]
        lab_name = f_files['lab'][cmg]

        lab_df = uro_utilities.read_uro_laborie(lab_name,
                                                columns=['EMGR2_A',
                                                         'Pabd',
                                                         'Pves',
                                                         'Pdet',
                                                         'Inf Vol',
                                                         'VH2O'],
                                                round=4).set_index('Time')

        uro_df = pd.read_csv(uro_name, delimiter='\t')

        cough_time = uro_utilities.get_cough_time(uro_name)

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

        # lab_df = lab_df.reset_index().rename(columns={'index': 'Time'})
        # lab_df.loc[:, 'Time'] = lab_df.loc[:, 'Time'].apply(pd.to_timedelta, unit='sec')

        void_emg = [lab_df.query('Time > @vtime - 5 and Time < @vtime + 5').loc[:, 'EMGR2_A'] for
                    vtime in void_times]
        void_p_n[study_letter][cmg] = []
        for vemg in void_emg:
            F_psi, cf, tres, bandwidth = calculate_wavelets(vemg.shape[0],
                                                            sampling_rate=1 / lab_df.index.values[
                                                                1])
            freqs = np.linspace(0, 500, vemg.shape[0])
            void_p_n[study_letter][cmg].append(pd.DataFrame(data=calculate_wavelet_spectrum(
                    vemg.values.ravel(),
                    F_psi,
                    len(F_psi[0, :]),
                    freqs,
                    cf,
                    tres),
                    index=vemg.index.to_series().apply(pd.to_timedelta, unit='sec'),
                    dtype=np.float64))

    return void_p_n
