import os
import pandas as pd
import numpy as np
import h5py
import json
from scipy.interpolate import interp1d
from customized_logger import logger as logging
from const import TRIALS, SUBJECTS, ALL_FIELDS, KAM_PHASE, FORCE_PHASE, STEP_PHASE
from const import SAMPLES_BEFORE_STEP, SAMPLES_AFTER_STEP, DATA_PATH, L_PLATE_FORCE_Z, FORCE_DATA_FIELDS
from const import R_PLATE_FORCE_Z, R_KAM_COLUMN, EVENT_COLUMN, CONTINUOUS_FIELDS, DISCRETE_FIELDS, SEGMENT_DEFINITIONS
from const import SEGMENT_DATA_FIELDS, FORCE_PHASE_THRESHOLD


def get_step_data(step_array):
    def get_segment_point(seg_axis):
        seg_name, axis = seg_axis[:-2], seg_axis[-1:]
        markers = SEGMENT_DEFINITIONS[seg_name]
        markers_in_one_direction = list(map(lambda x: x + '_' + axis, markers))
        return pd.DataFrame(step_array[markers_in_one_direction].mean(axis=1), columns=[seg_axis])

    step_array = pd.concat([step_array]+list(map(get_segment_point, SEGMENT_DATA_FIELDS)), axis=1)

    def get_one_step_data(_id):
        step = step_array[np.abs(step_array[EVENT_COLUMN]) == _id]
        min_index, max_index = step.index.min(), step.index.max()
        extended_step = step_array[min_index-SAMPLES_BEFORE_STEP:max_index+SAMPLES_AFTER_STEP]
        extended_step.index = range(extended_step.shape[0])
        max_shape = max_step_length+SAMPLES_BEFORE_STEP+SAMPLES_AFTER_STEP
        extended_step = extended_step.reindex(range(max_shape))
        extended_step[np.isnan(extended_step)] = 0.
        extended_step[STEP_PHASE] = np.where(np.abs(extended_step[EVENT_COLUMN]) == _id, 1., 0.)
        return extended_step

    target_ids = np.abs(step_array[EVENT_COLUMN]).drop_duplicates().dropna()
    return map(get_one_step_data, target_ids)


def append_force_phase(one_step):
    one_step[FORCE_PHASE] = np.where(one_step[R_PLATE_FORCE_Z][:] < -FORCE_PHASE_THRESHOLD, 1., 0.)
    return one_step


def fill_invalid_cop(one_step, safe_frame=3):
    step_length = np.sum(np.where((one_step.iloc[:, :10] == 0).all(axis=1), 0, 1))
    f_nonzero = np.where(one_step[FORCE_PHASE] == 1)[0]
    f_nonzero = f_nonzero[f_nonzero > 10]
    f_nonzero = f_nonzero[f_nonzero < 150]
    for name in FORCE_DATA_FIELDS[9:12]:
        one_step[name].iloc[:f_nonzero[safe_frame - 1]] = one_step[name].iloc[f_nonzero[safe_frame - 1]]
        one_step[name].iloc[f_nonzero[-safe_frame]:step_length] = one_step[name].iloc[f_nonzero[-safe_frame]]
    return one_step


def append_kam_phase(one_step):
    step_max_index = one_step[one_step[STEP_PHASE] == 1.].index.max()
    min_index, max_index = np.where(
        (one_step[R_PLATE_FORCE_Z] * one_step[STEP_PHASE] < -20.)[:step_max_index])[0][[0, -1]]
    mid_index = (min_index + max_index) // 2
    peak_index = one_step[R_PLATE_FORCE_Z].loc[:mid_index].idxmin()
    try:
        min_index = np.where(one_step.loc[:peak_index, R_KAM_COLUMN] < 0.)[0][-1]
    except IndexError:
        pass
    one_step[KAM_PHASE] = np.where(np.logical_and(min_index < one_step.index, one_step.index < max_index), 1., 0.)
    return one_step


def is_step_data_corrupted(one_step):
    return (one_step[EVENT_COLUMN] >= 0.).all()


def is_foot_on_right_plate_alone(one_step_array):
    min_index, max_index = one_step_array[one_step_array[FORCE_PHASE] == 1.].index[[0, -1]]
    left, right = int(0.7 * min_index + 0.3 * max_index), int(0.3 * min_index + 0.7 * max_index)
    return (one_step_array.loc[left:right, L_PLATE_FORCE_Z] > -20).any()


def is_kam_positive(one_step):
    min_index, max_index = one_step[one_step[FORCE_PHASE] == 1.].index[[0, -1]]
    mid_index = (min_index + max_index) // 2
    peak_index = one_step.loc[:mid_index, R_PLATE_FORCE_Z].idxmin()
    return (one_step.loc[peak_index:mid_index, R_KAM_COLUMN] > 0.).all()


def is_kam_length_reasonable(one_step):
    return np.ptp(np.where(one_step[KAM_PHASE] == 1.)) > 30


def resample_to_100_sample(one_step):
    x = np.linspace(0., 1., one_step.shape[0])
    new_x = np.linspace(0., 1., 100)
    f = interp1d(x, one_step[CONTINUOUS_FIELDS], axis=0, kind=3)
    g = interp1d(x, one_step[DISCRETE_FIELDS], axis=0, kind='nearest')
    return pd.DataFrame(np.concatenate([f(new_x), g(new_x)], axis=1), columns=ALL_FIELDS)


def generate_step_data(export_path, processes):
    export_path = os.path.join(DATA_PATH, export_path)
    # create training data and test data
    subject_trial_dict = {}
    for subject_trial, data in all_data_dict.items():
        subject_trial_dict[subject_trial] = get_step_data(data)
        for f in processes:
            subject_trial_dict[subject_trial] = f(subject_trial_dict[subject_trial])

    subject_data_dict = {subject: [data for trial in TRIALS for data in subject_trial_dict[subject + " " + trial]]
                         for subject in SUBJECTS}
    with h5py.File(export_path, 'w') as hf:
        for subject, data_collections in subject_data_dict.items():
            subject_whole_trial = np.concatenate(
                [np.array(data.loc[:, ALL_FIELDS])[np.newaxis, :, :] for data in data_collections], axis=0)
            hf.create_dataset(subject, data=subject_whole_trial, dtype='float32')
        hf.attrs['columns'] = json.dumps(ALL_FIELDS)


if __name__ == "__main__":
    logging.debug("Loading all csv data")
    all_data_dict = {subject + " " + trial: pd.read_csv(os.path.join(DATA_PATH, subject, "combined", trial + ".csv"))
                     for subject in SUBJECTS for trial in TRIALS}
    max_step_length = max([trial_data[EVENT_COLUMN].value_counts().max() for trial_data in all_data_dict.values()])
    custom_process = [
        lambda step_data_list: map(append_force_phase, step_data_list),
        lambda step_data_list: map(fill_invalid_cop, step_data_list),
        lambda step_data_list: map(append_kam_phase, step_data_list),
        lambda step_data_list: filter(is_step_data_corrupted, step_data_list),
        lambda step_data_list: filter(is_foot_on_right_plate_alone, step_data_list),
        lambda step_data_list: filter(is_kam_positive, step_data_list),
        lambda step_data_list: filter(is_kam_length_reasonable, step_data_list),
        # lambda step_data_list: map(resample_to_100_sample, step_data_list), # This will resample step data to the same length, but it was used but obsolete because the results are not better.
    ]
    generate_step_data('40samples+stance.h5', custom_process)

