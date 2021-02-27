import wearable_toolkit
import pandas as pd
import os
from const import SEGMENT_DEFINITIONS, SUBJECTS, TRIALS, DATA_PATH, SUBJECT_HEIGHT, SUBJECT_WEIGHT, \
    SUBJECT_ID, TRIAL_ID

subject_infos = pd.read_csv(os.path.join(DATA_PATH, 'subject_info.csv'), index_col=0)


def sync_and_crop_data_frame(subject, trial):
    # files location
    vicon_data_path = os.path.join(DATA_PATH, subject, 'vicon', trial + '.csv')
    vicon_calibrate_data_path = os.path.join(DATA_PATH, subject, 'vicon', 'calibrate' + '.csv')
    imu_data_path = os.path.join(DATA_PATH, subject, 'imu', trial + '.csv')
    middle_data_path = os.path.join(DATA_PATH, subject, 'combined', trial + '.csv')
    is_verbose = False

    # read vicon, imu data
    subject_info = subject_infos.loc[subject, :]
    vicon_data = wearable_toolkit.ViconCsvReader(vicon_data_path, SEGMENT_DEFINITIONS, vicon_calibrate_data_path, subject_info)
    imu_data = wearable_toolkit.SageCsvReader(imu_data_path)

    # create step events
    imu_data.create_step_id('R_FOOT', verbose=False)

    # Synchronize Vicon and IMU data
    vicon_sync_data = vicon_data.get_angular_velocity_theta('R_SHANK', 1000)
    imu_sync_data = imu_data.get_norm('R_SHANK', 'Gyro')[0:1000]
    print("vicon-imu synchronization")
    vicon_imu_sync_delay = wearable_toolkit.sync_via_correlation(vicon_sync_data, imu_sync_data, is_verbose)

    # crop redundant data
    minimum_delay = min([-imu_data.get_first_event_index(), vicon_imu_sync_delay])
    vicon_delay = 0 - minimum_delay
    imu_delay = vicon_imu_sync_delay - minimum_delay
    imu_data.crop(imu_delay)
    vicon_data.crop(vicon_delay)
    min_length = min([x.data_frame.shape[0] for x in [imu_data, vicon_data]])
    middle_data = pd.concat([imu_data.data_frame, vicon_data.data_frame], axis=1)
    middle_data = middle_data.loc[:min_length]

    # append static data
    middle_data[SUBJECT_HEIGHT] = subject_info[SUBJECT_HEIGHT]
    middle_data[SUBJECT_WEIGHT] = subject_info[SUBJECT_WEIGHT]
    middle_data[SUBJECT_ID] = SUBJECTS.index(subject)
    middle_data[TRIAL_ID] = TRIALS.index(trial)
    middle_data.to_csv(middle_data_path)


if __name__ == "__main__":
    for s in SUBJECTS:
        for t in TRIALS:
            print("Subject {}, Trial {}".format(s, t))
            sync_and_crop_data_frame(s, t)
