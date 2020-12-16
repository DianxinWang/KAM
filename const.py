import os

DATA_PATH = os.environ.get('KAM_DATA_PATH')
TRIALS = ['baseline', 'fpa', 'step_width', 'trunk_sway']
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
SENSOR_LIST = ['L_FOOT', 'R_FOOT', 'R_SHANK', 'R_THIGH', 'WAIST', 'CHEST', 'L_SHANK', 'L_THIGH']
IMU_FIELDS = ['AccelX', 'AccelY', 'AccelZ', 'GyroX', 'GyroY', 'GyroZ', 'MagX', 'MagY', 'MagZ', 'Quat1', 'Quat2',
              'Quat3', 'Quat4']

extract_imu_fields = lambda imus, fields: [field + "_" + imu for imu in imus for field in fields]
extract_video_fields = lambda videos, angles: [video + "_" + position + "_" + angle for video in videos
                                               for position in ["x", "y"] for angle in angles]
VIDEO_LIST = ["LShoulder", "RShoulder", "MidHip", "RHip", "LHip", "RKnee", "LKnee", "RAnkle", "LAnkle", "RHeel",
              "LHeel"]
VIDEO_ANGLES = ["90", "180"]

VIDEO_DATA_FIELDS = extract_video_fields(VIDEO_LIST, VIDEO_ANGLES)
IMU_DATA_FIELDS = extract_imu_fields(SENSOR_LIST, IMU_FIELDS)

SAMPLES_BEFORE_STEP = 40
PADDING_MODES = [PADDING_ZERO, PADDING_NEXT_STEP] = range(2)
PADDING_MODE = PADDING_ZERO

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

PHASE_LIST = [EVENT_COLUMN, KAM_PHASE, FORCE_PHASE, STEP_PHASE] = ['Event', 'kam_phase', 'force_phase', 'step_phase']
# all the fields of combined data
CONTINUOUS_FIELDS = TARGETS_LIST + EXT_KNEE_MOMENT + IMU_DATA_FIELDS + VIDEO_DATA_FIELDS + FORCE_DATA_FIELDS +\
                    JOINT_LIST + SEGMENT_DATA_FIELDS
DISCRETE_FIELDS = STATIC_DATA + PHASE_LIST
ALL_FIELDS = DISCRETE_FIELDS + CONTINUOUS_FIELDS
