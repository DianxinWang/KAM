"""
wearalbe_toolkit.py

@Author: Dianxin

This package is used to preprocess data collected from walking trials.
Read v3d exported csv data, sage csv data, vicon exported csv data and openPose exported csv data.
Synchronize vicon data and sage data.

"""
import csv
import numpy as np
import math
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import MinMaxScaler

from const import SENSOR_LIST, IMU_FIELDS, FORCE_DATA_FIELDS, EXT_KNEE_MOMENT, SUBJECT_HEIGHT, EVENT_COLUMN
from const import SUBJECT_WEIGHT
import wearable_math


[PARENT_TITLE, SAMPLE_RATE, TITLE, DIRECTION, UNITS, DATA] = range(6)


class ViconCsvReader:
    """
    Read the csv files exported from vicon.
    The csv file should only contain Trajectories information
    """

    # if static_trial is not none, it's used for filling missing data.
    def __init__(self, file_path, segment_definitions=None, static_trial=None, sub_info=None):
        self.data, self.sample_rate = ViconCsvReader.reading(file_path)
        # create segment marker data
        self.segment_data = dict()
        if segment_definitions is None:
            segment_definitions = {}
        for segment, markers in segment_definitions.items():
            self.segment_data[segment] = pd.Series(dict([(marker, self.data[marker]) for marker in markers]))

        # used for filling missing marker data
        if static_trial is not None:
            calibrate_data, _ = ViconCsvReader.reading(static_trial)
            for segment, markers in segment_definitions.items():
                segment_data = pd.Series(dict([(marker, calibrate_data[marker]) for marker in markers]))
                self.fill_missing_marker(segment_data, self.segment_data[segment])

        # filter and resample force data
        force_names_ori = ['Imported Bertec Force Plate #' + plate_num + ' - ' + data_type for plate_num in ['1', '2']
                           for data_type in ['Force', 'CoP']]
        filtered_force_array = np.concatenate([data_filter(self.data[force_name], 50, 1000)
                                               for force_name in force_names_ori], axis=1)
        filtered_force_array = filtered_force_array[::10, :]
        filtered_force_df = pd.DataFrame(filtered_force_array, columns=FORCE_DATA_FIELDS)
        cal_offset = sub_info[['Caliwand for plate 1-x', 'Caliwand for plate 1-y', 'Caliwand for plate 1-z',
                               'Caliwand for plate 2-x', 'Caliwand for plate 2-y', 'Caliwand for plate 2-z']]
        filtered_force_df[['plate_1_cop_x', 'plate_1_cop_y', 'plate_1_cop_z',
                           'plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']] += cal_offset.values

        self.segment_definitions = segment_definitions
        if segment_definitions != {}:
            markers = [marker for markers in segment_definitions.values() for marker in markers]
            self.data_frame = pd.concat([self.data[marker] for marker in markers], axis=1)
            self.data_frame.columns = [marker + '_' + axis for marker in markers for axis in ['X', 'Y', 'Z']]

        self.data_frame = pd.concat([self.data_frame, filtered_force_df], axis=1)
        knee_moment = self.get_right_external_kam(sub_info)
        self.data_frame = pd.concat([self.data_frame, knee_moment], axis=1)

    @staticmethod
    def reading(file_path):
        data_collection = dict()
        sample_rate_collection = {}
        state = PARENT_TITLE
        with open(file_path, encoding='utf-8-sig') as f:
            for row in csv.reader(f):
                if state == PARENT_TITLE:
                    parent_title = row[0]
                    state = SAMPLE_RATE
                elif state == SAMPLE_RATE:
                    sample_rate_collection[parent_title] = float(row[0])
                    state = TITLE
                elif state == TITLE:
                    titles = list()
                    for col in row[2:]:
                        if col != "":
                            if 'Trajectories' == parent_title:
                                subject_name, title = col.split(':')
                            else:
                                title = col
                        titles.append(title)
                    state = DIRECTION
                elif state == DIRECTION:
                    directions = [i for i in row[2:]]
                    data = [[] for _ in directions]
                    state = UNITS
                elif state == UNITS:
                    # TODO@Dianxin: Record the units! never do it.
                    state = DATA
                elif state == DATA:
                    if row == []:
                        state = PARENT_TITLE
                        for title in titles:
                            data_collection[title] = dict()
                        for i, direction in enumerate(directions):
                            data_collection[titles[i]][direction] = data[i]
                        for key, value in data_collection.items():
                            data_collection[key] = pd.DataFrame(value)
                        continue
                    for i, x in enumerate(row[2:]):
                        try:
                            data[i].append(float(x))
                        except ValueError:
                            data[i].append(np.nan)
        return data_collection, sample_rate_collection

    def get_right_external_kam(self, sub_info):
        sub_height, sub_weight = sub_info[[SUBJECT_HEIGHT, SUBJECT_WEIGHT]]
        # cal_offset = sub_info[['Caliwand for plate 2-x', 'Caliwand for plate 2-y', 'Caliwand for plate 2-z']]
        force_cop = self.data_frame[['plate_2_cop_x', 'plate_2_cop_y', 'plate_2_cop_z']].values
        # force_cop += cal_offset
        knee_origin = (self.data_frame[['RFME_X', 'RFME_Y', 'RFME_Z']].values +
                       self.data_frame[['RFLE_X', 'RFLE_Y', 'RFLE_Z']].values) / 2
        r = force_cop - knee_origin
        force_data = -self.data_frame[['plate_2_force_x', 'plate_2_force_y', 'plate_2_force_z']].values
        knee_moment = pd.DataFrame(np.cross(r, force_data), columns=EXT_KNEE_MOMENT)
        knee_moment /= (sub_height * sub_weight * 1000.)
        return knee_moment

    def get_angular_velocity_theta(self, segment, check_len):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']

        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        check_len = min(walking_data.shape[0], check_len)
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity_theta = np.zeros([check_len])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        # vectiorize this for loop.
        for i_frame in range(check_len):
            if i_frame == 0:
                continue
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            R_one_sample, _ = rigid_transform_3d(current_marker_matrix, next_marker_matrix)
            theta = np.math.acos((np.matrix.trace(R_one_sample) - 1) / 2)

            angular_velocity_theta[i_frame] = theta * sampling_rate / np.pi * 180
        return angular_velocity_theta

    def get_rshank_angle(self, direction):
        ankle = (self.data['RTAM'] + self.data['RFAL']) / 2
        knee = (self.data['RFME'] + self.data['RFLE']) / 2
        vector = (knee - ankle).values
        if direction == 'X':
            return np.arctan2(vector[:, 2], vector[:, 1])
        elif direction == 'Y':
            return np.arctan2(vector[:, 2], vector[:, 0])
        elif direction == 'Z':
            return np.arctan2(vector[:, 1], vector[:, 0])

    def get_angular_velocity(self, segment, direction):
        segment_data_series = self.segment_data[segment]
        sampling_rate = self.sample_rate['Trajectories']
        walking_data = pd.concat(segment_data_series.tolist(), axis=1).values
        data_len = walking_data.shape[0]
        marker_number = int(walking_data.shape[1] / 3)
        angular_velocity = np.zeros([data_len, 3])

        next_marker_matrix = walking_data[0, :].reshape([marker_number, 3])
        if direction == 'X':
            next_marker_matrix[:, 0] = 0
        elif direction == 'Y':
            next_marker_matrix[:, 1] = 0
        elif direction == 'Z':
            next_marker_matrix[:, 2] = 0
        # vectiorize this for loop.
        for i_frame in range(1, data_len):
            current_marker_matrix = next_marker_matrix
            next_marker_matrix = walking_data[i_frame, :].reshape([marker_number, 3])
            if direction == 'X':
                next_marker_matrix[:, 0] = 0
            elif direction == 'Y':
                next_marker_matrix[:, 1] = 0
            elif direction == 'Z':
                next_marker_matrix[:, 2] = 0
            R_one_sample, _ = rigid_transform_3d(current_marker_matrix, next_marker_matrix)
            theta = rotation_matrix_to_euler_angles(R_one_sample)

            angular_velocity[i_frame, :] = theta * sampling_rate / np.pi * 180
        angular_velocity = pd.DataFrame(angular_velocity)
        angular_velocity.columns = ['X', 'Y', 'Z']
        return angular_velocity[direction]

    def get_marker_position(self, marker_name):
        return self.data[marker_name]

    def crop(self, start_index):
        # keep index after start_index
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = range(self.data_frame.shape[0])

        for segment, markers in self.segment_definitions.items():
            for marker in markers:
                self.segment_data[segment][marker] = self.segment_data[segment][marker].loc[start_index:]
                self.segment_data[segment][marker].index = range(self.segment_data[segment][marker].shape[0])

    def fill_missing_marker(self, calibrate_makers, motion_markers):
        if sum([motion_marker.isnull().sum().sum() for motion_marker in motion_markers.tolist()]) == 0:
            return

        # take the first frame of calibrate marker data for calibration
        calibrate_makers = pd.concat(calibrate_makers.tolist(), axis=1).values
        calibrate_makers = calibrate_makers[0, :].reshape([-1, 3])

        walking_data = pd.concat(motion_markers.tolist(), axis=1).values
        data_len = walking_data.shape[0]

        for i_frame in range(data_len):
            marker_matrix = walking_data[i_frame, :].reshape([-1, 3])
            coordinate_points = np.argwhere(~np.isnan(marker_matrix[:, 0])).reshape(-1)
            missing_points = np.argwhere(np.isnan(marker_matrix[:, 0])).reshape(-1)
            if len(missing_points) == 0:  # All the marker exist
                continue
            if len(coordinate_points) >= 3:
                origin, x, y, z = wearable_math.generate_coordinate(calibrate_makers[coordinate_points, :])
                origin_m, x_m, y_m, z_m = wearable_math.generate_coordinate(marker_matrix[coordinate_points, :])
                for missing_point in missing_points.tolist():
                    relative_point = wearable_math.get_relative_position(origin, x, y, z,
                                                                         calibrate_makers[missing_point, :])
                    missing_data = wearable_math.get_world_position(origin_m, x_m, y_m, z_m, relative_point)
                    motion_markers[missing_point]['X'][i_frame] = missing_data[0]
                    motion_markers[missing_point]['Y'][i_frame] = missing_data[1]
                    motion_markers[missing_point]['Z'][i_frame] = missing_data[2]
        for motion_marker in motion_markers:
            motion_marker.interpolate(method='linear', axis=0, inplace=True)

    def append_external_kam(self):
        # calibrate force plate
        pass


class SageCsvReader:
    """
    Read the csv file exported from sage systems
    """
    GUESSED_EVENT_INDEX = 0

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.sample_rate = 100
        self.data_frame = self.data[
            [field + '_' + str(index) for index, label in enumerate(SENSOR_LIST) for field in
             IMU_FIELDS]].copy()
        index = self.data['Package_0']
        for i in range(1, len(self.data['Package_0'])):
            if self.data['Package_0'].loc[i] < self.data['Package_0'].loc[i - 1]:
                self.data.loc[i:, 'Package_0'] += 65536
        index = index - self.data['Package_0'].loc[0]
        # fill dropout data with nan
        if index.size - 1 != index.iloc[-1]:
            print("Inconsistent shape")
        self.data_frame.index = index
        self.data_frame = self.data_frame.reindex(range(0, int(index.iloc[-1] + 1)))
        self.data_frame.columns = ["_".join([col.split('_')[0], SENSOR_LIST[int(col.split('_')[1])]]) for col in
                                   self.data_frame.columns]
        self.missing_data_index = self.data_frame.isnull().any(axis=1)
        self.data_frame = self.data_frame.interpolate(method='linear', axis=0)
        # self.data_frame.loc[:, :] = data_filter(self.data_frame.values, 15, 100, 2)

    def get_norm(self, sensor, field, is_plot=False):
        assert sensor in SENSOR_LIST
        assert field in ['Accel', 'Gyro']
        norm_array = np.linalg.norm(self.data_frame[[field + direct + '_' + sensor for direct in ['X', 'Y', 'Z']]],
                                    axis=1)
        if is_plot:
            plt.figure()
            plt.plot(norm_array)
            plt.show()
        return norm_array

    def get_first_event_index(self):
        for i in range(len(self.data['sync_event'])):
            if self.data['sync_event'].loc[i] == 1:
                return i
        return None

    def get_field_data(self, sensor, field):
        if sensor not in SENSOR_LIST:
            raise RuntimeError("No such a sensor")
        if field not in ['Accel', 'Gyro']:
            raise RuntimeError("{field} not in ['Accel', 'Gyro']")
        index = str(SENSOR_LIST.index(sensor))
        data = self.data_frame[[field + direct + '_' + str(index) for direct in ['X', 'Y', 'Z']]]
        data.columns = ['X', 'Y', 'Z']
        return data

    def crop(self, start_index):
        self.data = self.data.loc[start_index:]
        self.data.index = self.data.index - self.data.index[0]
        self.data_frame = self.data_frame.loc[start_index:]
        self.data_frame.index = self.data_frame.index - self.data_frame.index[0]

    def get_walking_strike_off(self, strike_delay, off_delay, segment, cut_off_fre_strike_off=None,
                               verbose=False):
        """ Reliable algorithm used in TNSRE first submission"""
        gyr_thd = np.rad2deg(2.6)
        acc_thd = 1.2
        max_distance = self.sample_rate * 2  # distance from stationary phase should be smaller than 2 seconds

        acc_data = np.array(
            self.data_frame[['_'.join([direct, segment]) for direct in ['AccelX', 'AccelY', 'AccelZ']]])
        gyr_data = np.array(self.data_frame[['_'.join([direct, segment]) for direct in ['GyroX', 'GyroY', 'GyroZ']]])

        if cut_off_fre_strike_off is not None:
            acc_data = data_filter(acc_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)
            gyr_data = data_filter(gyr_data, cut_off_fre_strike_off, self.sample_rate, filter_order=2)

        gyr_x = gyr_data[:, 0]
        data_len = gyr_data.shape[0]

        acc_magnitude = np.linalg.norm(acc_data, axis=1)
        gyr_magnitude = np.linalg.norm(gyr_data, axis=1)
        acc_magnitude = acc_magnitude - 9.81

        stationary_flag = self.__find_stationary_phase(
            gyr_magnitude, acc_magnitude, acc_thd, gyr_thd)

        strike_list, off_list = [], []
        i_sample = 0

        while i_sample < data_len:
            # step 0, go to the next stationary phase
            if not stationary_flag[i_sample]:
                i_sample += 1
            else:
                front_crossing, back_crossing = self.__find_zero_crossing(gyr_x, gyr_thd, i_sample)

                if not back_crossing:  # if back zero crossing not found
                    break
                if not front_crossing:  # if front zero crossing not found
                    i_sample = back_crossing
                    continue

                the_strike = self.find_peak_max(gyr_x[front_crossing:i_sample], height=0)
                the_off = self.find_peak_max(gyr_x[i_sample:back_crossing], height=0)

                if the_strike is not None and i_sample - (the_strike + front_crossing) < max_distance:
                    strike_list.append(the_strike + front_crossing + strike_delay)
                if the_off is not None and the_off < max_distance:
                    off_list.append(the_off + i_sample + off_delay)
                i_sample = back_crossing
        if verbose:
            plt.figure()
            plt.plot(stationary_flag * 400)
            plt.plot(gyr_x)
            plt.plot(strike_list, gyr_x[strike_list], 'g*')
            plt.plot(off_list, gyr_x[off_list], 'r*')

        return strike_list, off_list

    @staticmethod
    def __find_stationary_phase(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        """ Old function, require 10 continuous setps """
        data_len = gyr_magnitude.shape[0]
        stationary_flag, stationary_flag_temp = np.zeros(gyr_magnitude.shape), np.zeros(gyr_magnitude.shape)
        stationary_flag_temp[
            (acc_magnitude < foot_stationary_acc_thd) & (abs(gyr_magnitude) < foot_stationary_gyr_thd)] = 1
        for i_sample in range(data_len):
            if stationary_flag_temp[i_sample - 5:i_sample + 5].all():
                stationary_flag[i_sample] = 1
        return stationary_flag

    @staticmethod
    def __find_stationary_phase_2(gyr_magnitude, acc_magnitude, foot_stationary_acc_thd, foot_stationary_gyr_thd):
        """ New function, removed 10 sample requirement """
        stationary_flag = np.zeros(gyr_magnitude.shape)
        stationary_flag[(acc_magnitude < foot_stationary_acc_thd) & (gyr_magnitude < foot_stationary_gyr_thd)] = 1
        return stationary_flag

    def __find_zero_crossing(self, gyr_x, foot_stationary_gyr_thd, i_sample):
        """
        Detected as a zero crossing if the value is lower than negative threshold.
        :return:
        """
        max_search_range = self.sample_rate * 3  # search 3 second front data at most
        front_crossing, back_crossing = False, False
        for j_sample in range(i_sample, max(0, i_sample - max_search_range), -1):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                front_crossing = j_sample
                break
        for j_sample in range(i_sample, gyr_x.shape[0]):
            if gyr_x[j_sample] < - foot_stationary_gyr_thd:
                back_crossing = j_sample
                break
        return front_crossing, back_crossing

    @staticmethod
    def find_peak_max(data_clip, height, width=None, prominence=None):
        """
        find the maximum peak
        :return:
        """
        peaks, properties = find_peaks(data_clip, width=width, height=height, prominence=prominence)
        if len(peaks) == 0:
            return None
        peak_heights = properties['peak_heights']
        max_index = np.argmax(peak_heights)
        return peaks[max_index]

    def create_step_id(self, segment, verbose=False):
        max_step_length = self.sample_rate * 2
        [RON, ROFF] = self.get_walking_strike_off(0, 0, segment, 10, verbose)
        events_dict = {'ROFF': ROFF, 'RON': RON}
        foot_events = translate_step_event_to_step_id(events_dict, max_step_length)
        self.data_frame.insert(0, EVENT_COLUMN, np.nan)
        if verbose:
            plt.figure()
        for _, event in foot_events.iterrows():
            self.data_frame.loc[event[0]:event[1], EVENT_COLUMN] = SageCsvReader.GUESSED_EVENT_INDEX
            SageCsvReader.GUESSED_EVENT_INDEX += 1
            if verbose:
                plt.plot(self.data_frame.loc[event[0]:event[1], 'GyroX_'+segment].values)
        if self.missing_data_index.any(axis=0):
            print("Steps containing corrupted data: {}. They are marked as minus".format(
                self.data_frame[self.missing_data_index][EVENT_COLUMN].dropna().drop_duplicates().tolist()))
            self.data_frame.loc[self.missing_data_index, EVENT_COLUMN] *= -1  # mark the missing IMU data as minus event
        if verbose:
            plt.show()


class DivideMaxScalar(MinMaxScaler):
    def partial_fit(self, X, y=None):
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        data_range = data_max - data_min
        data_bi_max = np.nanmax(abs(X), axis=0)
        self.scale_ = 1 / data_bi_max
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        X *= self.scale_
        return X


def data_filter(data, cut_off_fre, sampling_fre, filter_order=4):
    fre = cut_off_fre / (sampling_fre / 2)
    b, a = butter(filter_order, fre, 'lowpass')
    if len(data.shape) == 1:
        data_filtered = filtfilt(b, a, data)
    else:
        data_filtered = filtfilt(b, a, data, axis=0)
    return data_filtered


def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rigid_transform_3d(a, b):
    """
    Get the Rotation Matrix and Translation array between A and B.
    return:
        R: Rotation Matrix, 3*3
        T: Translation Array, 1*3
    """
    assert len(a) == len(b)

    N = a.shape[0]  # total points
    centroid_A = np.mean(a, axis=0)
    centroid_B = np.mean(b, axis=0)
    # centre the points
    AA = a - np.tile(centroid_A, (N, 1))
    BB = b - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    H = np.dot(AA.T, BB)
    U, _, V_t = linalg.svd(np.nan_to_num(H))
    R = np.dot(V_t.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        # print
        # "Reflection detected"
        V_t[2, :] *= -1
        R = np.dot(V_t.T, U.T)
    T = -np.dot(R, centroid_A.T) + centroid_B.T
    return R, T


def sync_via_correlation(data1, data2, verbose=False):
    correlation = np.correlate(data1, data2, 'full')
    delay = len(data2) - np.argmax(correlation) - 1
    if verbose:
        plt.figure()
        if delay > 0:
            plt.plot(data1)
            plt.plot(data2[delay:])
        else:
            plt.plot(data1[-delay:])
            plt.plot(data2)
        plt.show()
    return delay


def translate_step_event_to_step_id(events_dict, max_step_length):
    # FILTER EVENTS
    event_list = sorted(
        [[i, event_type] for event_type in ['RON', 'ROFF'] for i in events_dict[event_type]], key=lambda x: x[0])
    event_type_dict = {i: event_type for i, event_type in event_list}
    event_ids = [i[0] for i in event_list]
    RON_events = events_dict['RON']

    def is_qualified_ron_event(ron_i):
        i = event_ids.index(RON_events[ron_i])
        prev_event_type, curr_event_type, next_event_type = map(lambda x: event_type_dict[event_ids[x]], [i-1, i, i+1])
        prev_step_length, current_step_length = np.diff(RON_events[ron_i - 2:ron_i+1])
        if curr_event_type not in [prev_event_type, next_event_type]\
                and 1.33 * prev_step_length > current_step_length > 0.75 * prev_step_length\
                and 50 < current_step_length < max_step_length:
            return True
        return False

    def transform_to_step_events(ron_i):
        """return consecutive events: off, on, off, on"""
        current_event_id_i = event_ids.index(RON_events[ron_i])
        return map(lambda i: event_ids[i], range(current_event_id_i-3, current_event_id_i + 1))

    r_steps = filter(is_qualified_ron_event, range(10, len(RON_events)))
    r_steps = map(transform_to_step_events, r_steps)
    r_steps = pd.DataFrame(r_steps)
    r_steps.columns = ['off_3', 'on_2', 'off_1', 'on_0']
    stance_columns = ['on_2', 'off_1']
    return r_steps[stance_columns]


def calibrate_force_plate_center(file_path, plate_num):
    assert (plate_num in [1, 2])
    vicon_data = ViconCsvReader(file_path)
    data_DL = vicon_data.data['DL']
    data_DR = vicon_data.data['DR']
    data_ML = vicon_data.data['ML']
    center_vicon = (data_DL + data_DR) / 2 + (data_DL - data_ML)
    if plate_num == 1:
        center_plate = vicon_data.data['Imported Bertec Force Plate #1 - CoP']
    else:
        center_plate = vicon_data.data['Imported Bertec Force Plate #2 - CoP']
    center_plate.columns = ['X', 'Y', 'Z']
    plate_cop = np.mean(center_plate, axis=0)
    cop_offset = np.mean(center_vicon, axis=0) - plate_cop
    return plate_cop, cop_offset

