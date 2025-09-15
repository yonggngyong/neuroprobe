import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os, json

from .config import *
from .braintreebank_subject import BrainTreebankSubject

# Defining the names of evaluations and preparing them for downstream processing
single_float_variables_name_remapping = {
    "pitch": "enhanced_pitch", #"pitch",
    "volume": "rms", #"rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "delta_pitch": "delta_enhanced_pitch",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length"
}
four_way_cardinal_directions_name_remapping = {
    "global_flow_angle": "max_global_angle",
    "local_flow_angle": "max_vector_angle",
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos"
}
new_pitch_variables = ['enhanced_pitch', 'enhanced_volume', 'delta_enhanced_pitch', 'delta_enhanced_volume', 'raw_pitch', 'raw_volume', 'delta_raw_pitch', 'delta_raw_volume']
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys()) + new_pitch_variables
four_way_cardinal_direction_variables = list(four_way_cardinal_directions_name_remapping.values()) + list(four_way_cardinal_directions_name_remapping.keys())
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + four_way_cardinal_direction_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index", "speaker"] + classification_variables


class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name, output_indices=False, 
                 start_neural_data_before_word_onset=START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE, end_neural_data_after_word_onset=END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE,
                 lite=True, nano=False, random_seed=NEUROPROBE_GLOBAL_RANDOM_SEED, output_dict=False):
        """
        Args:
            subject (Subject): the subject to evaluate on
            trial_id (int): the trial to evaluate on
            dtype (torch.dtype): the data type of the returned data
            eval_name (str): the name of the variable to evaluate on
                Options for eval_name (from the Neuroprobe paper):
                    frame_brightness, global_flow, local_flow, global_flow_angle, local_flow_angle, face_num, volume, pitch, delta_volume, 
                    delta_pitch, speech, onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech, speaker
            lite (bool): if True, the eval is Neuroprobe-Lite (the default), otherwise it is Neuroprobe-Full
            nano (bool): if True, the eval is Neuroprobe-Nano (the default), otherwise it is Neuroprobe-Lite (if lite is True)

            output_indices (bool): 
                if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
                if False, the dataset will output the neural data directly

            output_dict (bool): 
                if True, the dataset will output a dictionary with the following keys:
                    "data": the neural data -- either directly or as a tuple (index_from, index_to)
                    "label": the label
                    "electrode_labels": the labels of the electrodes
                If False, the dataset will output a tuple (input, label) or ((index_from, index_to), label) directly
            
            start_neural_data_before_word_onset (int): the number of samples to start the neural data before each word onset
            end_neural_data_after_word_onset (int): the number of samples to end the neural data after each word onset
            random_seed (int): seed for random operations within this dataset
        """

        # Set up a local random state with the provided seed
        self.rng = np.random.RandomState(random_seed)
        
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.output_indices = output_indices
        self.start_neural_data_before_word_onset = start_neural_data_before_word_onset
        self.end_neural_data_after_word_onset = end_neural_data_after_word_onset
        self.lite = lite
        self.nano = nano
        self.n_classes = 2
        self.output_dict = output_dict

        if self.nano:
            nano_electrodes = NEUROPROBE_NANO_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in nano_electrodes if e in subject.electrode_labels]
            self.electrode_labels_subset = [e for e in nano_electrodes if e in subject.electrode_labels]
        elif self.lite:
            lite_electrodes = NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in lite_electrodes if e in subject.electrode_labels]
            self.electrode_labels_subset = [e for e in lite_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_NANO_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_NANO_SUBJECT_TRIALS"
        elif self.lite:
            lite_electrodes = NEUROPROBE_LITE_ELECTRODES[subject.subject_identifier]
            self.electrode_indices_subset = [subject.electrode_labels.index(e) for e in lite_electrodes if e in subject.electrode_labels]
            self.electrode_labels = [subject.electrode_labels[i] for i in self.electrode_indices_subset]
            subject_trial = (subject.subject_id, self.trial_id)
            assert subject_trial in NEUROPROBE_LITE_SUBJECT_TRIALS, f"Subject {subject.subject_id} trial {self.trial_id} not in NEUROPROBE_LITE_SUBJECT_TRIALS"
        else:
            # use all electrode labels and indices
            self.electrode_indices_subset = np.arange(len(subject.electrode_labels))
            self.electrode_labels = subject.electrode_labels
        self.electrode_coordinates = subject.get_electrode_coordinates()[self.electrode_indices_subset]

        eval_name_remapped = eval_name
        if eval_name in single_float_variables_name_remapping: eval_name_remapped = single_float_variables_name_remapping[eval_name]
        if eval_name in four_way_cardinal_directions_name_remapping: eval_name_remapped = four_way_cardinal_directions_name_remapping[eval_name]
        if eval_name in classification_variables_name_remapping: eval_name_remapped = classification_variables_name_remapping[eval_name]
        self.eval_name_remapped = eval_name_remapped

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)

        self.movie_name = BRAINTREEBANK_SUBJECT_TRIAL_MOVIE_NAME_MAPPING[f"{self.subject.subject_identifier}_{self.trial_id}"]
        
        # Add the original features from braintreebank to the all_words_df
        transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{self.movie_name}/features.csv')
        original_features_df = pd.read_csv(transcript_file_format.format(self.movie_name)).set_index('Unnamed: 0')
        # Add new columns from words_df using original_index mapping
        new_columns = [col for col in original_features_df.columns if col not in self.all_words_df.columns]
        for col in new_columns:
            self.all_words_df[col] = self.all_words_df['original_index'].map(original_features_df[col])
        
        if eval_name in single_float_variables:
            # Grab the new pitch volume features if they exist
            if self.eval_name_remapped in new_pitch_variables:
                pitch_volume_features_path = os.path.join(PITCH_VOLUME_FEATURES_DIR, f"{self.movie_name}_pitch_volume_features.json")
                with open(pitch_volume_features_path, 'r') as f:
                    raw_pitch_volume_features = json.load(f)

                TARGET_DP_FOR_KEYS = 5  # Standard number of decimal places
                normalized_pvf = {}
                for k_str, v_val in raw_pitch_volume_features.items():
                    k_float = float(k_str)
                    normalized_key = f"{k_float:.{TARGET_DP_FOR_KEYS}f}"
                    normalized_pvf[normalized_key] = v_val
                pitch_volume_features = normalized_pvf

                start_times = self.all_words_df['start'].to_list()
                all_labels = []
                for start_time_val in start_times:
                    lookup_key = f"{start_time_val:.{TARGET_DP_FOR_KEYS}f}"
                    label = pitch_volume_features[lookup_key][self.eval_name_remapped]
                    all_labels.append(label)
                all_labels = np.array(all_labels)
            else:
                all_labels = self.all_words_df[self.eval_name_remapped].to_numpy()

            # Get indices for words in top and bottom quartiles
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            self.positive_indices = np.where(label_percentiles > 0.75)[0]
            self.negative_indices = np.where(label_percentiles < 0.25)[0]
        elif eval_name in ["onset", "speech"]:
            self.positive_indices = np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.all_words_df))
            self.negative_indices = np.arange(len(self.nonverbal_df))
        elif eval_name in four_way_cardinal_direction_variables: 
            self.class_labels = np.zeros(len(self.all_words_df), dtype=int)
            angles = self.all_words_df[self.eval_name_remapped].to_numpy()
            cardinal_directions = np.array([0, 90, 180, 270])
            angles_expanded = angles[:, np.newaxis]
            distances = np.minimum(np.abs(angles_expanded - cardinal_directions),
                                360 - np.abs(angles_expanded - cardinal_directions))
            class_labels = np.argmin(distances, axis=1)
            self.positive_indices = np.where(class_labels == 0)[0]
            self.negative_indices = np.where(class_labels == 2)[0]
        elif eval_name == "face_num":
            face_nums = self.all_words_df["face_num"].to_numpy().astype(int)
            self.positive_indices = np.where(face_nums > 0)[0]
            self.negative_indices = np.where(face_nums == 0)[0]
        elif eval_name == "word_index":
            word_indices = self.all_words_df["idx_in_sentence"].to_numpy().astype(int)
            self.positive_indices = np.where(word_indices == 0)[0]
            self.negative_indices = np.where(word_indices == 1)[0]
        elif eval_name == "word_head_pos":
            head_pos = self.all_words_df[self.eval_name_remapped].to_numpy().astype(int)
            self.positive_indices = np.where(head_pos == 0)[0]
            self.negative_indices = np.where(head_pos == 1)[0]
        elif eval_name == "word_part_speech":
            pos = self.all_words_df[self.eval_name_remapped].to_numpy()         
            self.positive_indices = np.where(pos == "VERB")[0]
            self.negative_indices = np.where(pos == "NOUN")[0]
        elif eval_name == "speaker":
            speakers = self.all_words_df['speaker']
            most_frequent_speaker = speakers.value_counts().index[0]
            self.positive_indices = np.where(speakers == most_frequent_speaker)[0]
            self.negative_indices = np.where(speakers != most_frequent_speaker)[0]
        elif eval_name == "word_gap":
            word_gap_distribution = []
            for i in range(1, len(self.all_words_df)):
                if self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']: continue
                gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                word_gap_distribution.append(gap)
            word_gap_distribution = np.array(word_gap_distribution)

            positive_indices = []
            negative_indices = []
            for i in range(1, len(self.all_words_df)):
                if self.all_words_df.iloc[i]['sentence'] != self.all_words_df.iloc[i-1]['sentence']: continue
                gap = self.all_words_df.iloc[i]['start'] - self.all_words_df.iloc[i-1]['end']
                gap_percentile = np.mean(word_gap_distribution < gap)
                if gap_percentile > 0.75:
                    positive_indices.append(i)
                elif gap_percentile < 0.25:
                    negative_indices.append(i)
            self.positive_indices = np.array(positive_indices)
            self.negative_indices = np.array(negative_indices)
        else:
            raise ValueError(f"Invalid eval_name: {eval_name}")

        n_samples_each = min(len(self.positive_indices), len(self.negative_indices))
        if self.lite: 
            n_samples_each = min(n_samples_each, NEUROPROBE_LITE_MAX_SAMPLES//2)
        elif self.nano:
            n_samples_each = min(n_samples_each, NEUROPROBE_NANO_MAX_SAMPLES//2)
        self.positive_indices = np.sort(self.rng.choice(self.positive_indices, size=n_samples_each, replace=False))
        self.negative_indices = np.sort(self.rng.choice(self.negative_indices, size=n_samples_each, replace=False))
        self.n_samples = len(self.positive_indices) + len(self.negative_indices)
        

    def _get_neural_data(self, window_from, window_to, force_output_indices=False):
        self.subject.load_neural_data(self.trial_id)
        if not self.output_indices and not force_output_indices:
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=window_from, window_to=window_to)
            if self.lite or self.nano:
                input = input[self.electrode_indices_subset]
            return input.to(dtype=self.dtype)
        else:
            return window_from, window_to # just return the window indices

    def _positive_negative_getitem__(self, idx, force_output_indices=False):
        # even indices are positive samples, odd indices are negative samples
        word_index = self.positive_indices[idx//2] if idx % 2 == 0 else self.negative_indices[idx//2]
        if self.eval_name in ["onset", "speech"] and idx % 2 == 1: # for onset and speech, we need to get the nonverbal data
            row = self.nonverbal_df.iloc[word_index]
        else:
            row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(self.start_neural_data_before_word_onset)
        est_end_idx = est_idx + int(self.start_neural_data_before_word_onset) + int(self.end_neural_data_after_word_onset)
        input = self._get_neural_data(est_idx, est_end_idx, force_output_indices=force_output_indices)
        return input, (1 if idx % 2 == 0 else 0)
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx, force_output_indices=False):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.n_samples}")
        input, label = self._positive_negative_getitem__(idx, force_output_indices=force_output_indices)
        
        if self.output_dict:
            return {
                "data": input, 
                "label": label, 
                "electrode_labels_subset" : self.electrode_labels_subset,
                "electrode_labels": self.electrode_labels,
                "electrode_coordinates": self.electrode_coordinates,
                "metadata": {
                    "subject_identifier": self.subject.subject_identifier,
                    "trial_id": self.trial_id,
                    "sampling_rate": 2048,
                }
            }
        else:
            return input, label