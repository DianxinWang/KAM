import os

DATA_PATH = os.environ.get('KAM_DATA_PATH')
TRIALS = ['baseline', 'fpa', 'step_width', 'trunk_sway']
TRIALS_PRINT = ['Baseline', 'FPA', 'Step Width', 'Trunk Sway']
STATIC_TRIALS = ['static_back', 'static_side']
SUBJECTS = ['s002_wangdianxin', 's004_ouyangjue', 's005_tangansheng', 's006_xusen', 's007_zuogangao', 's008_liyu',
            's009_sunyubo', 's010_handai', 's011_wuxingze', 's012_likaixiang', 's013_zhangxiaohan', 's014_maqichao',
            's015_weihuan', 's017_tantian', 's018_wangmian', 's019_chenhongyuan', 's020_houjikang'
            # , 's003_linyuan', 's001_tantian', 's016_houjikang'
            ]
SEGMENT_DEFINITIONS = {
    'L_FOOT': ['LFCC', 'LFM5', 'LFM2'],
    'R_FOOT': ['RFCC', 'RFM5', 'RFM2'],
    'L_SHANK': ['LTAM', 'LFAL', 'LSK', 'LTT'],
    'R_SHANK': ['RTAM', 'RFAL', 'RSK', 'RTT'],
    'L_THIGH': ['LFME', 'LFLE', 'LTH', 'LFT'],
    'R_THIGH': ['RFME', 'RFLE', 'RTH', 'RFT'],
    'WAIST': ['LIPS', 'RIPS', 'LIAS', 'RIAS'],
    'CHEST': ['MAI', 'SXS', 'SJN', 'CV7', 'LAC', 'RAC']
}
SEGMENT_DATA_FIELDS = [seg_name + '_' + axis for axis in ['X', 'Y', 'Z'] for seg_name in SEGMENT_DEFINITIONS.keys()]
SEGMENT_MASS_PERCENT = {'L_FOOT': 1.37, 'R_FOOT': 1.37, 'R_SHANK': 4.33, 'R_THIGH': 14.16,
                        'WAIST': 11.17, 'CHEST': 15.96, 'L_SHANK': 4.33, 'L_THIGH': 14.16}      # 15.96 + 16.33
SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Quat1', 'Quat2',
              'Quat3', 'Quat4']

extract_imu_fields = lambda imus, fields: [field + "_" + imu for imu in imus for field in fields]

IMU_DATA_FIELDS = extract_imu_fields(SENSOR_LIST, IMU_FIELDS)

SAMPLES_BEFORE_STEP = 20
SAMPLES_AFTER_STEP = 20
FORCE_PHASE_THRESHOLD = 20

L_PLATE_FORCE_Z, R_PLATE_FORCE_Z = ['plate_1_force_z', 'plate_2_force_z']

TARGETS_LIST = R_KAM_COLUMN, _, _, _ = ["RIGHT_KNEE_ADDUCTION_MOMENT", "RIGHT_KNEE_FLEXION_MOMENT",
                                        "RIGHT_KNEE_ADDUCTION_ANGLE", "RIGHT_KNEE_ADDUCTION_VELOCITY"]
EXT_KFM, EXT_KAM, _ = EXT_KNEE_MOMENT = ['EXT_KM_X', 'EXT_KM_Y', 'EXT_KM_Z']

JOINT_LIST = [marker + '_' + axis for axis in ['X', 'Y', 'Z'] for marker in sum(SEGMENT_DEFINITIONS.values(), [])]

extract_right_force_fields = lambda types, axes: ['plate_2_' + data_type + '_' + axis
                                                  for data_type in types for axis in axes]
FORCE_DATA_FIELDS = ['plate_' + num + '_' + data_type + '_' + axis for num in ['1', '2']
                     for data_type in ['force', 'cop'] for axis in ['x', 'y', 'z']]

STATIC_DATA = SUBJECT_WEIGHT, SUBJECT_HEIGHT = ['body weight', 'body height']

PHASE_LIST = [EVENT_COLUMN, KAM_PHASE, FORCE_PHASE, STEP_PHASE, SUBJECT_ID, TRIAL_ID] = ['Event', 'kam_phase', 'force_phase', 'step_phase', 'subject_id', 'trial_id']
# all the fields of combined data
CONTINUOUS_FIELDS = TARGETS_LIST + EXT_KNEE_MOMENT + IMU_DATA_FIELDS  + FORCE_DATA_FIELDS +\
                    JOINT_LIST + SEGMENT_DATA_FIELDS
DISCRETE_FIELDS = STATIC_DATA + PHASE_LIST
ALL_FIELDS = DISCRETE_FIELDS + CONTINUOUS_FIELDS

FONT_SIZE_LARGE = 24
FONT_SIZE = 20
FONT_SIZE_SMALL = 16
FONT_DICT = {'fontsize': FONT_SIZE, 'fontname': 'DejaVu Sans'}
FONT_DICT_LARGE = {'fontsize': FONT_SIZE_LARGE, 'fontname': 'DejaVu Sans'}
FONT_DICT_SMALL = {'fontsize': FONT_SIZE_SMALL, 'fontname': 'DejaVu Sans'}
FONT_DICT_X_SMALL = {'fontsize': 15, 'fontname': 'DejaVu Sans'}
LINE_WIDTH = 2
LINE_WIDTH_THICK = 3

SENSOR_COMBINATION = ['8IMU_2camera', '8IMU', '3IMU_2camera', '3IMU', '1IMU_2camera', '1IMU', '2camera']
