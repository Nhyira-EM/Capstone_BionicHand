# import os
# import numpy as np
# import scipy.io
# from scipy.signal import resample_poly
# from tqdm import tqdm

# input_folder = r'C:\Users\nhyir\Downloads\gesture-recognition-and-biometrics-electromyogram-grabmyo-1.1.0\Output BM\Session1_converted'
# output_folder = "./ProcessedGRABMyo/Session1/"
# os.makedirs(output_folder, exist_ok=True)

# window_size = 200  # corresponds to 200ms
# sma_window = 20    # moving average filter length
# orig_fs = 2048
# target_fs = 1000
# downsample_factor = orig_fs / target_fs

# def preprocess_emg(signal):
#     # 1. Mean subtraction
#     signal = signal - np.mean(signal, axis=0)

#     # 2. Rectification
#     signal = np.abs(signal)

#     # 3. Moving average filtering
#     def moving_average(x, w):
#         return np.convolve(x, np.ones(w)/w, mode='same')
    
#     for i in range(signal.shape[1]):
#         signal[:, i] = moving_average(signal[:, i], sma_window)

#     # 4. Normalization (0 to 1)
#     signal_min = np.min(signal, axis=0)
#     signal_max = np.max(signal, axis=0)
#     signal = (signal - signal_min) / (signal_max - signal_min + 1e-8)

#     return signal

# for file in tqdm(os.listdir(input_folder)):
#     if file.endswith(".mat"):
#         mat_data = scipy.io.loadmat(os.path.join(input_folder, file))
#         forearm_data = mat_data["DATA_FOREARM"]

#         subject_id = os.path.splitext(file)[0]

#         for gesture_idx, cell in enumerate(forearm_data[0]):
#             emg_raw = cell  # shape: (10240, 16)
            
#             # Downsample
#             emg_downsampled = resample_poly(emg_raw, up=1000, down=2048, axis=0)

#             # Preprocess
#             emg_processed = preprocess_emg(emg_downsampled)

#             # Create overlapping or non-overlapping windows
#             num_samples = emg_processed.shape[0]
#             windows = []
#             for start in range(0, num_samples - window_size + 1, window_size):  # non-overlapping
#                 window = emg_processed[start:start + window_size, :]
#                 windows.append(window)

#             windows = np.stack(windows)  # shape: (num_windows, 200, 16)

#             # Save
#             output_path = os.path.join(output_folder, f"{subject_id}_gesture{gesture_idx+1}.npy")
#             np.save(output_path, windows)

import os
import numpy as np
import scipy.io
from scipy.signal import resample_poly
import pandas as pd
from tqdm import tqdm

# Paths
input_folder = r'C:\Users\nhyir\Downloads\gesture-recognition-and-biometrics-electromyogram-grabmyo-1.1.0\Output BM\Session1_converted'
output_folder = "./ProcessedGRABMyo/Session1_CSV/"
os.makedirs(output_folder, exist_ok=True)

# Settings
window_size = 200
sma_window = 20
orig_fs = 2048
target_fs = 1000
gesture_names = [
    "Index and Middle Finger Extension",
    "Little Finger Extension",
    "Hand Open",
    "Hand Close",
    "Rest"
]

def preprocess_emg(signal):
    signal = signal - np.mean(signal, axis=0)
    signal = np.abs(signal)

    def moving_average(x, w):
        return np.convolve(x, np.ones(w)/w, mode='same')

    for i in range(signal.shape[1]):
        signal[:, i] = moving_average(signal[:, i], sma_window)

    signal_min = np.min(signal, axis=0)
    signal_max = np.max(signal, axis=0)
    signal = (signal - signal_min) / (signal_max - signal_min + 1e-8)

    return signal

all_dataframes = []

for file in tqdm(os.listdir(input_folder)):
    if file.endswith(".mat"):
        mat_data = scipy.io.loadmat(os.path.join(input_folder, file))
        forearm_data = mat_data["DATA_FOREARM"]

        subject_id = os.path.splitext(file)[0]

        for gesture_idx, cell in enumerate(forearm_data[0][:5]):  # First 5 gestures only
            emg_raw = cell  # shape: (10240, 16)

            # Downsample
            emg_downsampled = resample_poly(emg_raw, up=1000, down=2048, axis=0)

            # Preprocess
            emg_processed = preprocess_emg(emg_downsampled)

            # Create 200-sample non-overlapping windows
            num_samples = emg_processed.shape[0]
            for start in range(0, num_samples - window_size + 1, window_size):
                window = emg_processed[start:start + window_size, :]  # shape: (200, 16)
                flat_window = window.flatten()  # shape: (3200,)

                # Add label and subject info
                row = np.append(flat_window, [gesture_names[gesture_idx], subject_id])
                all_dataframes.append(row)

# Convert to DataFrame
column_names = [f"ch{ch}_t{t}" for t in range(window_size) for ch in range(16)]
column_names += ["gesture", "subject"]
df = pd.DataFrame(all_dataframes, columns=column_names)

# Save to CSV
output_file = os.path.join(output_folder, "Processed_EMG_Windows.csv")
df.to_csv(output_file, index=False)

