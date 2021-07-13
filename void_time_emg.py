#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os
import roman
import uro_utilities
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from uro_utilities.mpl_figure_editor import MPLFigureEditor
from wavelets import calculate_wavelets, calculate_wavelet_spectrum

from traits.api import HasTraits, Dict, Instance
from traitsui.api import Group, Item, View, Action, Menu, MenuBar, CloseAction

from pyface.api import DirectoryDialog, OK


def fix_shape(table, resample_rate):
    for i, v in enumerate(table):
        if table[i].shape[1] != int(resample_rate[:-2]):
            table[i] = table[i][:, :-1]
    return table


class DContainer(HasTraits):
    """
    A :py:class:`traits.has_traits.HasTraits` wrapper class for dict instances.
    """
    value = Dict

    def __getattr__(self, key):
        if key in self.value:
            return self.value[key]

    def __setattr__(self, key, value):
        if key in self.value:
            self.value[key] = value
        else:
            super().__setattr__(key, value)


class VoidTimeEMGPlots(HasTraits):
    void_p_n = Dict
    dict_wrapper = Instance(DContainer)

    def __init__(self, void_p_n, **traits):
        self.void_p_n = void_p_n
        self.dict_wrapper = DContainer()
        self.dict_wrapper.value = {study_letter: DContainer() for study_letter in void_p_n.keys()}
        for study_letter in void_p_n.keys():
            self.dict_wrapper.value[study_letter].value = {cmg: Figure()
                                                           for cmg in void_p_n[study_letter].keys()}
        super().__init__(**traits)

    def create_plots(self):
        for study_letter in self.void_p_n.keys():
            for cmg in self.void_p_n[study_letter].keys():
                if len(self.void_p_n[study_letter][cmg]) == 0:
                    continue
                fig = self.dict_wrapper.value[study_letter].value[cmg]
                fig.supxlabel('Time')
                fig.supylabel('Wavelet Index')
                vmin = np.min([vemg.min() for vemg in self.void_p_n[study_letter][cmg]])
                vmax = np.max([vemg.max() for vemg in self.void_p_n[study_letter][cmg]])
                norm = Normalize(vmin=vmin, vmax=vmax)
                for i, vemg in enumerate(self.void_p_n[study_letter][cmg]):
                    ax = fig.add_subplot(1, len(self.void_p_n[study_letter][cmg]), i + 1)
                    emg_data = vemg.resample('100ms').mean().values.T
                    ax.pcolormesh(
                            vemg.resample('100ms').mean().index.values/pd.to_timedelta(1, 's'),
                            np.arange(emg_data.shape[0]),
                            emg_data,
                            shading='gouraud',
                            norm=norm
                    )
                    ax.set_title('Void Attempt {}'.format(i + 1))
                    # ax.set_yticks(np.arange(0.5, emg_data.shape[0] - 0.5))
                    # ax.set_yticklabels(np.arange(emg_data.shape[0] - 1))
                if vmin == vmax:
                    continue
                fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=fig.axes)
                fig.suptitle('EMG Analysis Near Void Attempts - {} {}'.format(study_letter, cmg))

    def save_all(self):
        dialog = DirectoryDialog()
        if dialog.open() == OK:
            for study_letter in self.void_p_n.keys():
                for cmg in self.void_p_n[study_letter].keys():
                    if len(self.void_p_n[study_letter][cmg]) == 0:
                        continue
                    self.dict_wrapper.value[study_letter].value[cmg].savefig(
                            os.path.join(dialog.path, '_'.join([study_letter, cmg])),
                            bbox_inches='tight')

    def default_traits_view(self):
        menubar = MenuBar(
                Menu(
                        Action(name='&Save All', action='save_all'),
                        CloseAction,
                        name='&File'
                )
        )
        items = Group(
                *[Group(
                        *[Group(
                                *[Item('object.dict_wrapper.{}.{}'.format(study_letter, cmg),
                                       label=cmg,
                                       editor=MPLFigureEditor(),
                                       show_label=False),
                                  ],
                                label='{}'.format(cmg)
                        ) for cmg in self.dict_wrapper.value[study_letter].value.keys()
                                if len(self.void_p_n[study_letter][cmg]) > 0
                        ],
                        label='{}'.format(study_letter),
                        layout='tabbed'
                ) for study_letter in self.dict_wrapper.value.keys()
                ], layout='tabbed'
        )
        return View(items, menubar=menubar, kind='modal', title='EMG Analysis at Void Attempts')


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
            if f.endswith(
                    '.txt') and 'Data Acquisition' in root and 'Laborie' in root and study_letter in f:
                found_files['lab'].append(os.path.join(root, f))
            if f.endswith('data.txt') and 'Tracking Files' in root and 'Data Acquisition' in root:
                found_files['uro'].append(os.path.join(root, f))

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
