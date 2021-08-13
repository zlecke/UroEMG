import matplotlib

matplotlib.rcParams['backend'] = 'Qt5Agg'
# matplotlib.rcParams['backend.qt4'] = 'PySide2'
# matplotlib.use('Qt5Agg')

import os
from traits.api import HasTraits, Instance, Directory, Bool, Button, Str, Int, observe
from traits_futures.api import TraitsExecutor, ProgressFuture, submit_progress
from traits_futures.future_states import FAILED
from traitsui.api import Item, Group, View, DirectoryEditor, InstanceEditor

from uro_utilities.study_list import StudyList
from uro_utilities.threading import QProgressEditor
from void_time_emg import void_time_wavelet
from uro_utilities.emg import UroEventEMGPlots


def analyze_emg(progress, study_list):
    output_p_n = {}
    output_cf = {}
    output_tres = {}
    output_bw = {}

    for i, study_path in enumerate(study_list.studies):
        study_letter = os.path.basename(study_path)[
                       os.path.basename(study_path).find(str(study_list.part_id)) + 3:
                       os.path.basename(study_path).find(str(study_list.part_id)) + 7]
        progress((i, len(study_list.studies), 'Analyzing {}'.format(study_letter)))
        p_n, center_freqs, time_res, bandwidths = void_time_wavelet(study_path,
                                                                    study_list.part_id,
                                                                    progress=progress)
        output_p_n = {**output_p_n, **p_n}
        output_cf = {**output_cf, **center_freqs}
        output_tres = {**output_tres, **time_res}
        output_bw = {**output_bw, **bandwidths}
    return output_p_n, output_cf, output_tres, output_bw


class UroEMG(HasTraits):
    prog_title = Str('Urodynamic EMG Analysis')

    executor = Instance(TraitsExecutor, ())

    future = Instance(ProgressFuture)
    study_list = Instance(StudyList, ())

    part_dir = Directory()
    _part_dir_label = Str('Participant Directory:')

    participant_selected = Bool(False)
    studies_selected = Bool(False)

    submit_find_studies = Button(label='Submit', style='button')

    submit_analyze_emg = Button(label='Submit', style='button')

    progress = Int(0)
    prog_message = Str()
    last_cur_step = Int(0)
    last_max_steps = Int(0)

    out_plots = Instance(UroEventEMGPlots)

    finished = Bool(False)

    # key_bindings = KeyBindings(KeyBinding())

    def default_traits_view(self):
        title_item = Item('prog_title',
                          show_label=False,
                          style='readonly',
                          style_sheet='*{qproperty-alignment:AlignHCenter; font-size: 24px; '
                                      'font-weight: bold}',
                          width=450
                          )
        opening_group = Group(
                Group(Item('_part_dir_label',
                           show_label=False,
                           style='readonly',
                           width=100),
                      Item('part_dir',
                           show_label=False,
                           editor=DirectoryEditor(entries=4, auto_set=True),
                           width=350),
                      orientation='horizontal'
                      ),
                Item('_', show_label=False),
                Group(Item('375'),
                      Item('submit_find_studies',
                           show_label=False,
                           width=25,
                           enabled_when='part_dir != ""'),
                      orientation='horizontal'),
                visible_when='not participant_selected and not studies_selected'
        )

        choose_studies_group = Group(
                Item('study_list', editor=InstanceEditor(), style='custom', show_label=False,
                     height=200),
                Item('_', show_label=False),
                Group(Item('375'),
                      Item('submit_analyze_emg', show_label=False, width=25),
                      orientation='horizontal'),
                visible_when='participant_selected and not studies_selected and not finished'
        )

        show_progress_group = Group(
                Item('progress',
                     show_label=False,
                     editor=QProgressEditor(min=0,
                                            max=100,
                                            show_percent=True,
                                            message_name='prog_message'),
                     style='simple'),
                visible_when='participant_selected and studies_selected and not finished'
        )

        return View(title_item,
                    opening_group,
                    choose_studies_group,
                    show_progress_group,
                    title='Urodynamic EMG Analysis',
                    resizable=True)

    def _submit_find_studies_fired(self):
        if not self.participant_selected:
            part_dir = self.part_dir
            self.study_list.part_id = int(os.path.basename(part_dir)[:3])
            self.study_list.find_studies(part_dir, True, False)
            self.participant_selected = True

    def _submit_analyze_emg_fired(self):
        self.future = submit_progress(self.executor, analyze_emg, study_list=self.study_list)
        self.studies_selected = True

    @observe('future:progress')
    def _report_progress(self, progress_info):
        if len(progress_info.new) == 3:
            cur_step, max_steps, msg = progress_info.new
            self.progress = int((cur_step / max_steps) * 100)
            self.prog_message = msg
            self.last_cur_step = cur_step
            self.last_max_steps = max_steps
        elif len(progress_info.new) == 2:
            frac, sub_message = progress_info.new
            tmp = frac * (1/self.last_max_steps) * 100
            self.progress = int(((self.last_cur_step/self.last_max_steps) * 100) + tmp)
            self.prog_message = " - ".join([self.prog_message.split(' - ')[0], sub_message])

    @observe('future:done')
    def _report_result(self, event):
        if self.future.state == FAILED:
            print(self.future.exception[2])
        else:
            self.out_plots = UroEventEMGPlots(*self.future.result)
            self.out_plots.create_plots()
            self.out_plots.edit_traits()
            self.finished = True


if __name__ == '__main__':
    app = UroEMG()
    x = app.configure_traits()
