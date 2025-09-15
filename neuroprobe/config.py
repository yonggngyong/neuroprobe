##########################
# 1. Paths and directories settings
##########################

import os

# It is recommended that all the paths are absolute!
# Feel free to change this setting to your own directory.
ROOT_DIR = "/global/cfs/cdirs/m4727/DIVER/DIVER_FINETUNING_DATASET/BrainTreebank"

# This comes together with the neuroprobe package
SAVE_SUBJECT_TRIAL_DF_DIR = "/global/cfs/cdirs/m4727/DIVER/neuroprobe/neuroprobe/braintreebank_features_time_alignment"
PITCH_VOLUME_FEATURES_DIR = "/global/cfs/cdirs/m4727/DIVER/neuroprobe/neuroprobe/neuroprobe/pitch_volume_features"

# Disable file locking for HDF5 files. This is helpful for parallel processing.
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE" 

##########################
# 2. Neuroprobe settings
# Do not change the settings below if you want the neuroprobe results to be compatible with others' evaluations!
##########################

SAMPLING_RATE = 2048 # Sampling rate for the BrainTreebank data (do not change this)
# Define a global random seed for reproducibility
NEUROPROBE_GLOBAL_RANDOM_SEED = 42  

# No need to change the settings below because you can change these values dynamically when defining the splits using the functions from neuroprobe_train_test_splits.py and the dataset from neuroprobe_datasets.py
START_NEURAL_DATA_BEFORE_WORD_ONSET = 0 # in seconds. NOTE: for the 1-second evaluation on the leaderboard, this is overridden to 0.
END_NEURAL_DATA_AFTER_WORD_ONSET = 1 # in seconds. NOTE: for the 1-second evaluation on the leaderboard, this is overridden to 1.
NEURAL_DATA_NONVERBAL_WINDOW_PADDING_TIME = 2 # how many seconds to wait between the last word off-set and the start of a "non-verbal" chunk
NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP = 0.5 # proportion of overlap between consecutive nonverbal chunks (0 means no overlap)
# some sanity check code as well as disabling file locking for HDF5 files
assert NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP >= 0 and NEURAL_DATA_NONVERBAL_WINDOW_OVERLAP < 1, "NONVERBAL_CONSECUTIVE_CHUNKS_OVERLAP must be between 0 and 1, strictly below 1"

# Standardizing pretraining and evaluation subjects and trials
DS_DM_TRAIN_SUBJECT_ID, DS_DM_TRAIN_TRIAL_ID = 2, 4
# Define the maximum number of samples to use for the lite datasetq
NEUROPROBE_LITE_MAX_SAMPLES = 3500
NEUROPROBE_LITE_N_FOLDS = 2
NEUROPROBE_LITE_SUBJECT_TRIALS = [
    (1, 1), (1, 2), 
    (2, 0), (2, 4),
    (3, 0), (3, 1),
    (4, 0), (4, 1),
    (7, 0), (7, 1),
    (10, 0), (10, 1)
]

NEUROPROBE_NANO_MAX_SAMPLES = 1000
NEUROPROBE_NANO_N_FOLDS = 2
NEUROPROBE_NANO_SUBJECT_TRIALS = [
    (1, 1), 
    (2, 4),
    (3, 1),
    (4, 0),
    (7, 1),
    (10, 1)
]

NEUROPROBE_TASKS = [
    "frame_brightness",
    "global_flow",
    "local_flow", 
    "global_flow_angle",
    "local_flow_angle",
    "face_num",
    "volume",
    "pitch",
    "delta_volume", 
    "delta_pitch",
    "speech",
    "onset",
    "gpt2_surprisal",
    "word_length",
    "word_gap",
    "word_index",
    "word_head_pos",
    "word_part_speech",
    "speaker"
]
NEUROPROBE_TASKS_MAPPING = {
    'onset': 'Sentence Onset',
    'speech': 'Speech',
    'volume': 'Volume', 
    'pitch': 'Voice Pitch',
    'speaker': 'Speaker Identity',
    'delta_volume': 'Delta Volume',
    'delta_pitch': 'Delta Pitch',
    'gpt2_surprisal': 'GPT-2 Surprisal',
    'word_length': 'Word Length',
    'word_gap': 'Inter-word Gap',
    'word_index': 'Word Position',
    'word_head_pos': 'Head Word Position',
    'word_part_speech': 'Part of Speech',
    'frame_brightness': 'Frame Brightness',
    'global_flow': 'Global Optical Flow',
    'local_flow': 'Local Optical Flow',
    'global_flow_angle': 'Global Flow Angle',
    'local_flow_angle': 'Local Flow Angle',
    'face_num': 'Number of Faces',
}

# Only used for the "Full" Neuroprobe dataset
NEUROPROBE_FULL_SUBJECT_TRIALS = [
    (1, 0), (1, 1), (1, 2), 
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 0), (3, 1), (3, 2), 
    (4, 0), (4, 1), (4, 2), 
    (5, 0), 
    (6, 0), (6, 1), (6, 4),
    (7, 0), (7, 1), 
    (8, 0), 
    (9, 0), 
    (10, 0), (10, 1)
]
NEUROPROBE_LONGEST_TRIALS_FOR_SUBJECT = {
    1: [0, 1],
    2: [4, 6],
    3: [2, 1],
    4: [2, 1],
    5: [0],
    6: [0, 2],
    7: [1, 0],
    8: [0], 
    9: [0],
    10: [1, 0]
}

BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING = {
    "btbank1_0": "fantastic-mr-fox",
    "btbank1_1": "the-martian",
    "btbank1_2": "thor-ragnarok",
    "btbank2_0": "venom",
    "btbank2_1": "spider-man-3-homecoming",
    "btbank2_2": "guardians-of-the-galaxy",
    "btbank2_3": "guardians-of-the-galaxy-2",
    "btbank2_4": "avengers-infinity-war",
    "btbank2_5": "black-panther",
    "btbank2_6": "aquaman",
    "btbank3_0": "cars-2",
    "btbank3_1": "lotr-1",
    "btbank3_2": "lotr-2",
    "btbank4_0": "shrek-the-third",
    "btbank4_1": "megamind",
    "btbank4_2": "incredibles",
    "btbank5_0": "fantastic-mr-fox",
    "btbank6_0": "megamind",
    "btbank6_1": "toy-story",
    "btbank6_4": "coraline",
    "btbank7_0": "cars-2",
    "btbank7_1": "megamind",
    "btbank8_0": "sesame-street-episode-3990",
    "btbank9_0": "ant-man",
    "btbank10_0": "cars-2",
    "btbank10_1": "spider-man-far-from-home"
}

NEUROPROBE_LITE_ELECTRODES = {"btbank1": ["T1bIc1", "T1bIc2", "T1bIc3", "T1bIc4", "T1bIc5", "T1bIc6", "T1bIc7", "T1bIc8", "T1cIf10", "T1cIf11", "T1cIf12", "T1cIf13", "T1cIf14", "T1cIf15", "T1cIf16", "T1aIb1", "T1aIb2", "T1aIb3", "T1aIb4", "T1aIb5", "T1aIb6", "T1aIb7", "T1aIb8", "T3aHb9", "T3aHb10", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5", "T1cIf6", "T1cIf7", "T1cIf8", "T2bHa7", "T2bHa8", "T2bHa9", "T2bHa10", "T2bHa11", "T2bHa12", "T2bHa13", "T2bHa14", "T3bOT8", "T3bOT9", "T3bOT10", "F3cId1", "F3cId2", "F3cId3", "F3cId4", "F3cId5", "F3cId6", "F3cId7", "F3cId8", "F3cId9", "T2c4", "T2c5", "T2c6", "T2c7", "T2c8", "F3bIaOFb1", "F3bIaOFb2", "F3bIaOFb3", "F3bIaOFb4", "F3bIaOFb5", "F3bIaOFb6", "F3bIaOFb7", "F3bIaOFb8", "F3bIaOFb9", "F3bIaOFb10", "F3bIaOFb11", "F3bIaOFb12", "F3bIaOFb13", "F3bIaOFb14", "F3bIaOFb15", "F3bIaOFb16", "T2d1", "T2d2", "T2d3", "T2d4", "T2d5", "T2d6", "F3aOFa7", "F3aOFa8", "F3aOFa9", "F3aOFa10", "F3aOFa11", "F3aOFa12", "F3aOFa13", "F3aOFa14", "F3aOFa15", "F3aOFa16", "F3aOFa2", "F3aOFa3", "F3aOFa4", "T3bOT1", "T3bOT2", "T3bOT3", "T3bOT4", "T3bOT5", "T3bOT6", "T2bHa3", "T2bHa4", "T2bHa5", "F3dIe1", "F3dIe2", "F3dIe3", "F3dIe4", "F3dIe5", "F3dIe6", "F3dIe7", "F3dIe8", "F3dIe9", "F3dIe10", "T2aA1", "T2aA2", "T2aA3", "T2aA4", "T2aA5", "T2aA6", "T2aA7", "T2aA8"], "btbank2": ["LT3d1", "LT3d2", "LT3d3", "LT3d4", "LT3d5", "LT3d6", "LT3d7", "LT3d8", "LT1c2", "LT1c3", "LT1c4", "LT1c5", "LT1c6", "LT1c7", "LT1c8", "LT1bIb1", "LT1bIb2", "LT1bIb3", "LT1bIb4", "LT1bIb5", "LT1bIb6", "LT1bIb7", "LT1bIb8", "LT1bIb9", "LT1bIb10", "RT1c7", "RT1c8", "RT1c9", "RT1c10", "RT1aIa1", "RT1aIa2", "RT1aIa3", "RT1aIa4", "RT1aIa5", "RT1aIa6", "RT1aIa7", "RT1aIa8", "RT1bIb1", "RT1bIb2", "RT1bIb3", "RT1bIb4", "RT1bIb5", "RT1bIb6", "RT1bIb7", "RT1bIb8", "RT1c1", "RT1c2", "RT1c3", "RT1c4", "RT1c5", "RT2aA1", "RT2aA2", "RT2aA3", "RT2aA4", "RT2aA5", "RT2aA6", "RT2aA7", "RT2aA8", "RT2aA9", "RT2aA10", "LT2aA4", "LT2aA5", "LT2aA6", "LT2aA7", "LT2aA8", "LT2aA9", "LT2aA10", "LT2aA11", "LT2aA12", "LT2aA13", "LT2aA14", "LT2aA1", "LT2aA2", "LT3a1", "LT3a2", "LT3a3", "LT3a4", "LT3a5", "LT3a6", "LT3a7", "LT3a8", "LT3a9", "LT3a10", "RT2b1", "RT2b2", "RT2b3", "RT2b4", "RT2b5", "LT3cHb4", "LT3cHb5", "LT3cHb6", "LT3cHb7", "LT3cHb8", "LT3cHb9", "LT3cHb10", "LT3cHb11", "LT3cHb12", "RT3bHb4", "RT3bHb5", "RT3bHb6", "RT3bHb7", "RT3bHb8", "RT3aHa4", "RT3aHa5", "RT3aHa6", "RT3aHa7", "RT3aHa8", "RT3aHa9", "RT3aHa10", "RT3aHa11", "RT3aHa12", "RT3aHa13", "RT3aHa14", "LT2b1", "LT2b2", "LT2b3", "LT2b4", "LT2b5", "LT2b6"], "btbank3": ["T1cIe5", "T1cIe6", "T1cIe7", "T1cIe8", "T1cIe9", "T1cIe10", "T1cIe11", "T1cIe12", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6", "T1aIc3", "T1aIc4", "T1aIc5", "T1aIc6", "O1aIb9", "O1aIb10", "O1aIb11", "O1aIb12", "O1aIb13", "O1aIb14", "O1aIb15", "O1aIb16", "F3d1", "F3d2", "F3d3", "F3d4", "F3d5", "F3d6", "F3d7", "F3d8", "F3d9", "F3d10", "F3aOF1", "F3aOF2", "F3aOF3", "F3aOF4", "F3aOF5", "F3aOF6", "F3aOF7", "F3aOF8", "F3c1", "F3c2", "F3c3", "F3c4", "F3c5", "F3c6", "F2Ia1", "F2Ia2", "F2Ia3", "F2Ia4", "F2Ia5", "F2Ia6", "F2Ia7", "F2Ia8", "F2Ia9", "F2Ia10", "F2Ia11", "F2Ia12", "F2Ia13", "F2Ia14", "P2a1", "P2a2", "P2a3", "P2a4", "P2a5", "P2a6", "P2a7", "P2a8", "P2a9", "O1bId8", "O1bId9", "O1bId10", "O1bId11", "O1bId12", "O1bId13", "O1bId14", "O1bId15", "O1bId16", "F3b7", "F3b8", "O1bId1", "O1bId2", "O1bId3", "O1bId4", "O1bId5", "F3b1", "F3b2", "F3b3", "F3b4", "P2b3", "P2b4", "P2b5", "P2b6", "P2b7", "P2b8", "P2b9", "P2b10", "P2b11", "P2b12", "P2b13", "P2b14", "P2b15", "P2b16", "T1cIe1", "T1cIe2"], "btbank4": ["LT1bId1", "LT1bId2", "LT1bId3", "LT1bId4", "LT1bId5", "LT1bId6", "LT1bId7", "LT1cCd1", "LT1cCd2", "LT1cCd3", "LT1cCd4", "LT1cCd5", "LT1cCd6", "LT1cCd7", "LT1cCd8", "LT1cCd9", "LT1cCd10", "LT1cCd11", "LT1cCd12", "LT1cCd13", "LT1cCd14", "LT1cCd15", "LT1cCd16", "LT3Ha1", "LT3Ha2", "LT3Ha3", "LT3Ha4", "LT3Ha5", "LT3Ha6", "LT3Ha7", "LT3Ha8", "LT3Ha9", "LF3cIc1", "LF3cIc2", "LF3cIc3", "LF3cIc4", "LF3cIc5", "LF3cIc6", "LF3cIc7", "LF3cIc8", "LF3cIc9", "LF3cIc10", "LF3bIa1", "LF3bIa2", "LF3bIa3", "LF3bIa4", "LF3bIa5", "LF3bIa6", "LF3bIa7", "LF3bIa8", "LF3bIa9", "LF3bIa10", "LF3bIa11", "LF3aOFa1", "LF3aOFa2", "LF3aOFa3", "LF3aOFa4", "LF3aOFa5", "LF3aOFa6", "LF3aOFa7", "LF3aOFa8", "LF3aOFa9", "LF3aOFa10", "LF3aOFa11", "LF3aOFa12", "LF3aOFa13", "LF3aOFa14", "LF3aOFa15", "LF3aOFa16", "LT2bHb1", "LT2bHb2", "LT2bHb3", "LT2bHb4", "LT2bHb5", "LT2bHb6", "LT2bHb7", "LT2bHb8", "LT2bHb9", "LT2bHb10", "LT2bHb11", "LT2bHb12", "LT2aA1", "LT2aA2", "LT2aA3", "LT2aA4", "LT2aA5", "LT2aA6", "LT2aA7", "LT2aA8", "LT2aA9", "LT2aA10", "LT2aA11", "LT2aA12", "LF1aCaOF1", "LF1aCaOF2", "LF1aCaOF3", "LF1aCaOF4", "LF1aCaOF5", "LF1aCaOF6", "LF1aCaOF7", "LF1aCaOF8", "LF1aCaOF9", "LF1aCaOF10", "LF1aCaOF11", "LF1aCaOF12", "LF1aCaOF13", "LF1aCaOF14", "LF1aCaOF15", "LF1aCaOF16", "LT1aIb1", "LT1aIb2", "LT1aIb3", "LT1aIb4", "LT1aIb5", "LT1aIb6", "LT1aIb7", "LT1aIb8", "LT1aIb9", "RF1cCc1", "RF1cCc2"], "btbank5": ["T1aA1", "T1aA2", "T1aA3", "T1aA4", "T1aA5", "T1aA6", "T1aA7", "T1aA8", "T1aA9", "T1aA10", "T1aA11", "T1aA12", "T1aA13", "T1aA14", "T1bId1", "T1bId2", "T1bId3", "T1bId4", "T1bId5", "T1bId6", "T1bId7", "T1bId8", "T1bId9", "T1bId10", "T1bId11", "T1bId12", "T1bId13", "T1bId14", "T1bId15", "T1bId16", "T2bHb5", "T2bHb6", "T2bHb7", "T2bHb8", "T2bHb9", "T2bHb10", "T2bHb11", "T2bHb12", "T2aHa4", "T2aHa5", "T2aHa6", "T2aHa7", "T2aHa8", "T2aHa9", "T2aHa10", "T2aHa11", "T2aHa12", "P2Ie2", "P2Ie3", "P2Ie4", "P2Ie5", "P2Ie6", "P2Ie7", "P2Ie8", "P2Ie9", "P2Ie10", "P2Ie11", "P2Ie12", "P2Ie13", "P2Ie14", "P2Ie15", "P2Ie16", "F3IaOF7", "F3IaOF8", "F3IaOF9", "F3IaOF10", "F3IaOF11", "F3IaOF12", "F3IaOF13", "F3IaOF14", "F3IaOF15", "F3IaOF16", "F2c2", "F2c3", "F2c4", "F1aCa1", "F1aCa2", "F1aCa3", "F1aCa4", "F1aCa5", "F1aCa6", "F1aCa7", "F1aCa8", "F1aCa9", "F1aCa10", "F1aCa11", "F1aCa12", "P1Cc1", "P1Cc2", "P1Cc3", "P1Cc4", "P1Cc5", "P1Cc6", "P1Cc7", "P1Cc8", "P1Cc9", "P1Cc10", "P1Cc11", "F2aIb1", "F2aIb2", "F2aIb3", "F2aIb4", "F2aIb5", "F2aIb6", "F2aIb7", "F2aIb8", "F2aIb9", "F2aIb10", "F1bCb1", "F1bCb2", "F1bCb3", "F1bCb4", "F1bCb5", "F1bCb6", "F1bCb7", "F1bCb8", "F1bCb9", "F1bCb10", "F1bCb11", "F1bCb12"], "btbank6": ["T1Id1", "T1Id2", "T1Id3", "T1Id4", "T1Id5", "T1Id6", "T1Id7", "T1Id8", "T2A8", "T2A9", "T2A10", "T2A11", "T2A12", "T2A13", "T2A14", "P2Ie1", "P2Ie2", "P2Ie3", "P2Ie4", "P2Ie5", "P2Ie6", "P2Ie7", "P2Ie8", "P2Ie9", "P2Ie10", "F3bOFb1", "F3bOFb2", "F3bOFb3", "F3bOFb4", "F3bOFb5", "F3bOFb6", "F3bOFb7", "F3bOFb8", "F3bOFb9", "F3bOFb10", "F3bOFb11", "F3bOFb12", "F3bOFb13", "F3bOFb14", "F3bOFb15", "F3bOFb16", "F3cOFc1", "F3cOFc2", "F3cOFc3", "F3cOFc4", "F3cOFc5", "F3cOFc6", "F3cOFc7", "F3cOFc8", "F3cOFc9", "F3cOFc10", "F3cOFc11", "F3cOFc12", "F3cOFc13", "F3cOFc14", "F3cOFc15", "F3cOFc16", "F3aOFa1", "F3aOFa2", "F3aOFa3", "F3aOFa4", "F3aOFa5", "F3aOFa6", "F3aOFa7", "F3aOFa8", "F3aOFa9", "F3aOFa10", "F3aOFa11", "F3aOFa12", "F3aOFa13", "F3aOFa14", "F3dIb1", "F3dIb2", "F3dIb3", "F3dIb4", "F3dIb5", "F3dIb6", "F3dIb7", "F3dIb8", "F3dIb9", "F3dIb10", "F2bIa1", "F2bIa2", "F2bIa3", "F2bIa4", "F2bIa5", "F2bIa6", "F2bIa7", "F2bIa8", "F2bIa9", "F2bIa10", "F2bIa11", "F2bIa12", "F2bIa13", "F2bIa14", "F2bIa15", "F2bIa16", "F2aCa1", "F2aCa2", "F2aCa3", "F2aCa4", "F2aCa5", "F2aCa6", "F2aCa7", "F2aCa8", "F2aCa9", "F2aCa10", "F2aCa11", "F2aCa12", "F2aCa13", "F2aCa14", "F2c1", "F2c2", "F2c3", "F2c4", "F2c5", "F2c6", "F2c7", "F2c8", "F2c9"], "btbank7": ["LT1A1", "LT1A2", "LT1A3", "LT1A4", "LT1A5", "LT1A6", "LT1A7", "LT1A8", "LT1A9", "LT1A10", "LT1A11", "LT1A12", "LT1A13", "LT1A14", "LF1aCaOF1", "LF1aCaOF2", "LF1aCaOF3", "LF1aCaOF4", "LF1aCaOF5", "LF1aCaOF6", "LF1aCaOF7", "LF1aCaOF8", "LF1aCaOF9", "LF1aCaOF10", "LF1aCaOF11", "LF1aCaOF12", "LF1aCaOF13", "LF1aCaOF14", "LF1aCaOF15", "LF1aCaOF16", "LT2H1", "LT2H2", "LT2H3", "LT2H4", "LT2H5", "LT2H6", "LT2H7", "LT2H8", "LT2H9", "LT2H10", "LT2H11", "LT2H12", "LT2H13", "LT2H14", "LT2H15", "LT2H16", "RF1aCaOF1", "RF1aCaOF2", "RF1aCaOF3", "RF1aCaOF4", "RF1aCaOF5", "RF1aCaOF6", "RF1aCaOF7", "RF1aCaOF8", "RF1aCaOF9", "RF1aCaOF10", "RF1aCaOF11", "RF1aCaOF12", "RF1aCaOF13", "RF1aCaOF14", "RF1aCaOF15", "RF1aCaOF16", "LF1bIb1", "LF1bIb2", "LF1bIb3", "LF1bIb4", "LF1bIb5", "LF1bIb6", "LF1bIb7", "LF1bIb8", "LF1bIb9", "LF1bIb10", "LF1bIb11", "LF1bIb12", "LF1bIb13", "LF1bIb14", "LF1bIb15", "LF1bIb16", "LF1cCb1", "LF1cCb2", "LF1cCb3", "LF1cCb4", "LF1cCb5", "LF1cCb6", "LF1cCb7", "LF1cCb8", "LF1cCb9", "LF1cCb10", "LF1cCb11", "LF2b1", "LF2b2", "LF2b3", "LF2b4", "LF2b5", "LF2b6", "LF2b7", "LF2b8", "LF2b9", "LF2b10", "LF2b11", "LF2b12", "LF2b13", "LF2b14", "LF3bOFb1", "LF3bOFb2", "LF3bOFb3", "LF3bOFb4", "LF3bOFb5", "LF3bOFb6", "LF3bOFb7", "LF3bOFb8", "LF3bOFb9", "LF3bOFb10", "LF3bOFb11", "LF3bOFb12", "LF3bOFb13", "LF3bOFb14", "LF3bOFb15", "LF3bOFb16"], "btbank8": ["T1bIF1", "T1bIF2", "T1bIF3", "T1bIF4", "T1bIF5", "T1bIF6", "T1bIF7", "T1bIF8", "F3cIc1", "F3cIc2", "F3cIc3", "F3cIc4", "F3cIc5", "F3cIc6", "F3cIc7", "F3cIc8", "T1aId1", "T1aId2", "T1aId3", "T1aId4", "T1aId5", "T1aId6", "T1aId7", "T1aId8", "T3H3", "T3H4", "T3H5", "T3H6", "T3H7", "T3H8", "T3H9", "T3H10", "T3H11", "T3H12", "F2bCb7", "F2bCb8", "F2bCb9", "F2bCb10", "F2bCb11", "F2bCb12", "F2bCb13", "T2A1", "T2A2", "T2A3", "T2A4", "F3bIaOFc1", "F3bIaOFc2", "F3bIaOFc3", "F3bIaOFc4", "F3bIaOFc5", "F3bIaOFc6", "F3bIaOFc7", "F3bIaOFc8", "F3bIaOFc9", "F3bIaOFc10", "F3bIaOFc11", "F3bIaOFc12", "F3bIaOFc13", "F3bIaOFc14", "F3bIaOFc15", "F3bIaOFc16", "T2A9", "T2A10", "T2A11", "T2A12", "T2A13", "T2A14", "F2cCc1", "F2cCc2", "F2cCc3", "F2cCc4", "F2cCc5", "F2cCc6", "F2cCc7", "F2cCc8", "F2cCc9", "F2cCc10", "F2cCc11", "F2cCc12", "F2cCc13", "F2cCc14", "F2aIb13", "F2aIb14", "F2aIb15", "F2aIb16", "F1aCaOFb1", "F1aCaOFb2", "F1aCaOFb3", "F1aCaOFb4", "F1aCaOFb5", "F1aCaOFb6", "F1aCaOFb7", "F1aCaOFb8", "F1aCaOFb9", "F1aCaOFb10", "F1aCaOFb11", "F1aCaOFb12", "F1aCaOFb13", "F1aCaOFb14", "F1aCaOFb15", "F1aCaOFb16", "F1bCd1", "F1bCd2", "F1bCd3", "F1bCd4", "F1bCd5", "F1bCd6", "F1bCd7", "F1bCd8", "F1bCd9", "F1bCd10", "F1bCd11", "F1bCd12", "P2Ie1", "P2Ie2", "P2Ie3", "P2Ie4", "P2Ie5", "P2Ie6", "P2Ie7"], "btbank9": ["F3b1", "F3b2", "F3b3", "F3b4", "F3b5", "F3b6", "T1aI1", "T1aI2", "T1aI3", "T1aI4", "T1aI5", "T1aI6", "T1aI7", "T1aI8", "F3a5", "F3a6", "P2b1", "P2b2", "P2b3", "P2b4", "P2b5", "P2b6", "P2b7", "P2b8", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6", "P2c6", "P2c7", "P2c8", "P2c9", "P2c10", "P2a1", "P2a2", "P2a3", "P2a4", "P2a5", "T1d1", "T1d2", "T1d3", "T1d4", "T1d5", "T1d6", "T1d7", "T1d8", "T1d9", "T1d10", "T1d11", "T1d12", "P2d1", "P2d2", "P2d3", "P2d4", "P2d5", "P2d6", "P2d7", "P2d8", "P2d9", "P2d10", "P2d11", "P2d12", "P2d13", "P2d14", "T1c1", "T1c2", "T1c3", "T1c4", "T1c5", "P2e1", "P2e2", "P2e3", "P2e4", "P2e5", "P2e6", "P2e7", "P2e8", "P1C1", "P1C2", "P1C3", "P1C4", "P1C5", "P1C6", "P1C7", "P1C8", "P1C9", "P1C10", "P1C11", "P1C12", "P1C13", "P1C14"], "btbank10": ["T1bId1", "T1bId2", "T1bId3", "T1bId4", "T1bId5", "T1bId6", "T1bId7", "T1bId8", "T1bId9", "T1bId10", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5", "T1cIf6", "T1cIf7", "T1cIf8", "T1cIf9", "T1cIf10", "T1cIf11", "T1cIf12", "T1aIa5", "T1aIa6", "T1aIa7", "T1aIa8", "T1aIa9", "T1aIa10", "F10Fa14", "F10Fa15", "F10Fa16", "F2e1", "F2e2", "F2e3", "F2e4", "F2e5", "F2e6", "F2e7", "F2e8", "T2aA1", "T2aA2", "T2aA3", "T2aA4", "F3bIb1", "F3bIb2", "F3bIb3", "F3bIb4", "F3bIb5", "F3bIb6", "F3bIb7", "F3bIb8", "F3bIb9", "F3bIb10", "F10Fa1", "F10Fa2", "F10Fa3", "F10Fa4", "P2aIe1", "P2aIe2", "P2aIe3", "P2aIe4", "P2aIe5", "P2aIe6", "P2aIe7", "P2aIe8", "P2aIe9", "P2aIe10", "P2aIe11", "P2aIe12", "F3cIc1", "F3cIc2", "F3cIc3", "F3cIc4", "F3cIc5", "F3cIc6", "F3cIc7", "F3cIc8", "F3cIc9", "F3cIc10", "T2aA7", "T2aA8", "T2aA9", "T2aA10", "T2aA11", "T2aA12", "T2aA13", "T2aA14", "T1aIa1", "T1aIa2", "T1aIa3", "F3aOFc1", "F3aOFc2", "F3aOFc3", "F3aOFc4", "F3aOFc5", "F3aOFc6", "F3aOFc7", "F3aOFc8", "F3aOFc9", "F3aOFc10", "F3aOFc11", "F3aOFc12", "F3aOFc13", "F3aOFc14", "F3aOFc15", "F3aOFc16", "P2cCc1", "P2cCc2", "P2cCc3", "P2cCc4", "T2bH1", "T2bH2", "P2cCc6", "P2cCc7", "P2cCc8", "P2cCc9", "P2cCc10", "P2cCc11", "P2cCc12", "P2cCc13"]}
NEUROPROBE_NANO_ELECTRODES = {"btbank1": ["T1bIc1", "T1bIc2", "T1bIc3", "T1bIc4", "T1bIc5", "T1bIc6", "T1bIc7", "T1bIc8", "T1cIf10", "T1cIf11", "T1cIf12", "T1cIf13", "T1cIf14", "T1cIf15", "T1cIf16", "T1aIb1", "T1aIb2", "T1aIb3", "T1aIb4", "T1aIb5", "T1aIb6", "T1aIb7", "T1aIb8", "T3aHb9", "T3aHb10", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5"], "btbank2": ["LT3d1", "LT3d2", "LT3d3", "LT3d4", "LT3d5", "LT3d6", "LT3d7", "LT3d8", "LT1c2", "LT1c3", "LT1c4", "LT1c5", "LT1c6", "LT1c7", "LT1c8", "LT1bIb1", "LT1bIb2", "LT1bIb3", "LT1bIb4", "LT1bIb5", "LT1bIb6", "LT1bIb7", "LT1bIb8", "LT1bIb9", "LT1bIb10", "RT1c7", "RT1c8", "RT1c9", "RT1c10"], "btbank3": ["T1cIe5", "T1cIe6", "T1cIe7", "T1cIe8", "T1cIe9", "T1cIe10", "T1cIe11", "T1cIe12", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6", "T1aIc3", "T1aIc4", "T1aIc5", "T1aIc6", "O1aIb9", "O1aIb10", "O1aIb11", "O1aIb12", "O1aIb13", "O1aIb14", "O1aIb15", "O1aIb16", "F3d1", "F3d2", "F3d3", "F3d4"], "btbank4": ["LT1bId1", "LT1bId2", "LT1bId3", "LT1bId4", "LT1bId5", "LT1bId6", "LT1bId7", "LT1cCd1", "LT1cCd2", "LT1cCd3", "LT1cCd4", "LT1cCd5", "LT1cCd6", "LT1cCd7", "LT1cCd8", "LT1cCd9", "LT1cCd10", "LT1cCd11", "LT1cCd12", "LT1cCd13", "LT1cCd14", "LT1cCd15", "LT1cCd16", "LT3Ha1", "LT3Ha2", "LT3Ha3", "LT3Ha4", "LT3Ha5", "LT3Ha6", "LT3Ha7"], "btbank5": ["T1aA1", "T1aA2", "T1aA3", "T1aA4", "T1aA5", "T1aA6", "T1aA7", "T1aA8", "T1aA9", "T1aA10", "T1aA11", "T1aA12", "T1aA13", "T1aA14", "T1bId1", "T1bId2", "T1bId3", "T1bId4", "T1bId5", "T1bId6", "T1bId7", "T1bId8", "T1bId9", "T1bId10", "T1bId11", "T1bId12", "T1bId13", "T1bId14", "T1bId15", "T1bId16"], "btbank6": ["T1Id1", "T1Id2", "T1Id3", "T1Id4", "T1Id5", "T1Id6", "T1Id7", "T1Id8", "T2A8", "T2A9", "T2A10", "T2A11", "T2A12", "T2A13", "T2A14", "P2Ie1", "P2Ie2", "P2Ie3", "P2Ie4", "P2Ie5", "P2Ie6", "P2Ie7", "P2Ie8", "P2Ie9", "P2Ie10", "F3bOFb1", "F3bOFb2", "F3bOFb3", "F3bOFb4", "F3bOFb5"], "btbank7": ["LT1A1", "LT1A2", "LT1A3", "LT1A4", "LT1A5", "LT1A6", "LT1A7", "LT1A8", "LT1A9", "LT1A10", "LT1A11", "LT1A12", "LT1A13", "LT1A14", "LF1aCaOF1", "LF1aCaOF2", "LF1aCaOF3", "LF1aCaOF4", "LF1aCaOF5", "LF1aCaOF6", "LF1aCaOF7", "LF1aCaOF8", "LF1aCaOF9", "LF1aCaOF10", "LF1aCaOF11", "LF1aCaOF12", "LF1aCaOF13", "LF1aCaOF14", "LF1aCaOF15", "LF1aCaOF16"], "btbank8": ["T1bIF1", "T1bIF2", "T1bIF3", "T1bIF4", "T1bIF5", "T1bIF6", "T1bIF7", "T1bIF8", "F3cIc1", "F3cIc2", "F3cIc3", "F3cIc4", "F3cIc5", "F3cIc6", "F3cIc7", "F3cIc8", "T1aId1", "T1aId2", "T1aId3", "T1aId4", "T1aId5", "T1aId6", "T1aId7", "T1aId8", "T3H3", "T3H4", "T3H5", "T3H6", "T3H7", "T3H8"], "btbank9": ["F3b1", "F3b2", "F3b3", "F3b4", "F3b5", "F3b6", "T1aI1", "T1aI2", "T1aI3", "T1aI4", "T1aI5", "T1aI6", "T1aI7", "T1aI8", "F3a5", "F3a6", "P2b1", "P2b2", "P2b3", "P2b4", "P2b5", "P2b6", "P2b7", "P2b8", "T1b1", "T1b2", "T1b3", "T1b4", "T1b5", "T1b6"], "btbank10": ["T1bId1", "T1bId2", "T1bId3", "T1bId4", "T1bId5", "T1bId6", "T1bId7", "T1bId8", "T1bId9", "T1bId10", "T1cIf1", "T1cIf2", "T1cIf3", "T1cIf4", "T1cIf5", "T1cIf6", "T1cIf7", "T1cIf8", "T1cIf9", "T1cIf10", "T1cIf11", "T1cIf12", "T1aIa5", "T1aIa6", "T1aIa7", "T1aIa8", "T1aIa9", "T1aIa10", "F10Fa14", "F10Fa15"]}