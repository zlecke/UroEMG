import os

from traits_futures.api import submit_progress

import uro_utilities
from uro_utilities import parse_cmg

from uro_utilities.wavelets import wavelet_analysis_near

from main import UroEMG


def analyze_storage(progress, study_list):
    output_p_n = {}
    output_cf = {}
    output_tres = {}
    output_bw = {}

    for i, study_path in enumerate(study_list.studies):
        study_letter = os.path.basename(study_path)[
                       os.path.basename(study_path).find(str(study_list.part_id)) + 3:
                       os.path.basename(study_path).find(str(study_list.part_id)) + 7]

        progress((i, len(study_list.studies), 'Analyzing {}'.format(study_letter)))
        stor_p_n = {study_letter: {}}
        cf = {study_letter: {}}
        time_res = {study_letter: {}}
        bandwidths = {study_letter: {}}

        f_files = {'lab': {}, 'comments': {}}

        for root, dirs, files in os.walk(study_path):
            for f in files:
                if f.endswith('.txt') \
                        and 'Data Acquisition' in root \
                        and 'Laborie' in root \
                        and study_letter in f:
                    f_files['lab'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)
                if f.endswith('_comments.txt') and ('pre-processing' == os.path.basename(root).lower()
                                                    or
                                                    'pre processing' == os.path.basename(root).lower()):
                    f_files['comments'][parse_cmg(os.path.join(root, f))] = os.path.join(root, f)

        for cmg in sorted(f_files['lab'].keys()):
            lab_name = f_files['lab'][cmg]

            m_vol_time, m_vol = uro_utilities.find_max_volume(lab_name)

            p_n, cf_i, time_res_i, bandwidths_i = wavelet_analysis_near(lab_name, [m_vol_time - 5])

            stor_p_n[study_letter][cmg] = p_n
            cf[study_letter][cmg] = cf_i
            time_res[study_letter][cmg] = time_res_i
            bandwidths[study_letter][cmg] = bandwidths_i

        output_p_n = {**output_p_n, **stor_p_n}
        output_cf = {**output_cf, **cf}
        output_tres = {**output_tres, **time_res}
        output_bw = {**output_bw, **bandwidths}
    return output_p_n, output_cf, output_tres, output_bw


class UroEMG_Storage(UroEMG):
    def _submit_analyze_emg_fired(self):
        self.future = submit_progress(self.executor, analyze_storage, study_list=self.study_list)
        self.studies_selected = True


if __name__ == '__main__':
    app = UroEMG_Storage()
    app.configure_traits()
