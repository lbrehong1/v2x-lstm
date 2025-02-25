#%% md
# # This is a sample Jupyter Notebook
# 
# Below is an example of a code cell. 
# Put your cursor into the cell and press Shift+Enter to execute it and select the next one, or click 'Run Cell' button.
# 
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 
# To learn more about Jupyter Notebooks in PyCharm, see [help](https://www.jetbrains.com/help/pycharm/ipython-notebook-support.html).
# For an overview of PyCharm, go to Help -> Learn IDE features or refer to [our documentation](https://www.jetbrains.com/help/pycharm/getting-started.html).
#%%
import time

from dotenv import load_dotenv
load_dotenv()

import argparse
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import keras
import tensorflow.keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from collections import deque
from tqdm import tqdm
import os
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#%%
#LOGFILE_PATH = "~/Documents/icccn-lstm/logs_cohda/trim_dsrc.txt"
#MODEL_PATH = "lstm_model.keras"

TIMESTEPS = 10
FEATURES = 6
EPOCHS = 5
SAMPLES = 20000 # number of samples to extract from the log file
TRAIN_RATIO = 0.1 # ratio of samples to use for training TODO this used to be 0.4
BATCH_SIZE = 32 # batch size for training, e.g. what is the number of samples to use for each epoch
VALIDATION_SPLIT = 0.15
MIN_LAT, MAX_LAT = 43.554669, 43.568290
MIN_LON, MAX_LON = 1.463952, 1.472176
MIN_LATENCY, MAX_LATENCY = 4,1000 # in ms
MIN_THROUGHPUT, MAX_THROUGHPUT = 0, 100 # in Mbps
MIN_PDR, MAX_PDR = 0, 1
#MIN_SINR, MAX_SINR =
MIN_RSRP, MAX_RSRP = -150, -45
TX_INTERVAL_MS = 20 # in ms, 50 packets per second
#TX_INTERVAL_MS = 100 # in ms, 10 packets per second
PDR_WINDOW = 1 # in seconds

#FEATURE_COLS_PC5 = ['tx_seq_num','tx_timestamp_ms','tx_latitude','tx_longitude','latency_ms','pdr']
#FEATURE_COLS_DSRC = ['tx_seq_num','tx_timestamp_ms','tx_latitude','tx_longitude','rsrp_1','rsrp_2','latency_ms','pdr']
FEATURE_COLS_PC5 = ['tx_latitude','tx_longitude','latency_ms','pdr']
FEATURE_COLS_DSRC = ['tx_latitude','tx_longitude','rsrp_1','rsrp_2','latency_ms','pdr']
#TARGET_COLS = ['throughput', 'pdr']
TARGET_COLS = ['latency_ms', 'pdr']

#%%
### Define pre-processing functions
def rmse(y_true, y_pred):
    y_true = K.cast(y_true, np.float32) # Ensure same type because auto-casting pulls float64
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_latest_model(rat):
    """
    Finds the most recent model file based on the RAT type.
    :param rat: RAT type ('pc5' or 'dsrc').
    :return: Path to the most recent model file.
    """
    model_files = glob.glob("model_" + rat + "_" + "*.keras")
    model_files_with_time = []
    regex = re.compile(r"model_" + rat + r"_(\d+).keras")
    for file in model_files:
        match = regex.search(os.path.basename(file))
        if match:
            model_files_with_time.append((file, int(match.group(1))))
    if not model_files_with_time:
        print("No existing models found.")
        return None
    # Sort by extracted timecode (newest first)
    latest_model = max(model_files_with_time, key=lambda x: x[1])[0]
    print("Found a model:", latest_model)
    return latest_model

def compute_pdr_rolling(df, time_column, window_size=1, packet_interval_ms=20):
    """
    Computes rolling PDR based on a fixed packet transmission rate.

    :param df: DataFrame containing the timestamps of received packets.
    :param time_column: Column name for timestamps.
    :param window_size: Time window (in seconds) for rolling PDR computation.
    :param packet_interval_ms: Fixed interval at which packets are sent (default: 20ms).
    :return: PDR rolling average series.
    """
    # Convert timestamps to seconds
    df["tx_timestamp_sec"] = df[time_column] / 1000
    # Compute rolling window size in terms of expected packets
    expected_packets_per_window = window_size * (1000 / packet_interval_ms)
    timestamps = df["tx_timestamp_sec"].to_numpy(dtype=float)
    rolling_window = deque()
    pdr_values = np.zeros(len(df)) # Ensure same length as df

    for i,t in enumerate(timestamps):
        # Remove timestamps outside the 1s window
        while rolling_window and rolling_window[0] < t - 1:
            rolling_window.popleft()
        # Add current timestamp
        rolling_window.append(t)
        if len(rolling_window) > expected_packets_per_window:
            #print(f"Warning: PDR value may exceed 1.0 at index {i} ({pdr_values[i]:.4f}). Popping leftmost value in window.")
            rolling_window.popleft()
        # Compute PDR
        pdr_values[i] = len(rolling_window) / expected_packets_per_window
        if pdr_values[i] > 1.0:
            print(f"Warning: PDR value still exceeds 1.0 at index {i} ({pdr_values[i]:.4f})")
            pdr_values[i] = 1.0

    new_column = pd.Series(pdr_values, index=df.index)
    try :
        df["pdr"] = new_column
    except ValueError as e:
        if len(df) != len(new_column):
            print("Length mismatch")
    return df

#%%
### Generate dummy data for demonstration (testing purposes)
# def generate_initial_data(samples=SAMPLES, timesteps=TIMESTEPS, features=FEATURES):
#     X_train = np.random.random((samples, timesteps, features))
#     y_train = np.random.random((samples,2))
#     return X_train, y_train
#
# # Initial training data
# X_train_generated, y_train_generated = generate_initial_data()
# print(f"X_train shape: {X_train_generated.shape}, y_train shape: {y_train_generated.shape}")

# data normalizer
def normalizer_init():
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler

# Normalize data. Requires an array of single values, except for GPS that requires [latitude, longitude] elements
def normalize(scaler, data):
    return scaler.transform(data)

def preprocess_lstm_input(df, rat, target_cols, time_column="tx_timestamp_ms", window_size_sec=PDR_WINDOW, packet_interval_ms=TX_INTERVAL_MS, seq_length=TIMESTEPS, train_ratio=TRAIN_RATIO):
    """
    Prepares LSTM input by computing PDR, normalizing features, and creating time series sequences.

    :param rat: RAT type ('pc5' or 'dsrc').
    :param df: DataFrame with raw network data.
    :param rat: RAT for this LSTM, used to determine feature column names.
    :param target_cols: List of target column names (e.g., ['throughput', 'PDR']).
    :param time_column: Name of timestamp column.
    :param window_size_sec: Time window for PDR computation.
    :param packet_interval_ms: Fixed packet send interval.
    :param seq_length: Number of past timesteps to use as input for LSTM.
    :param train_ratio: Ratio of training data to total data.
    """
    # Initialize variables
    gps_flag = False
    if rat == "pc5":
        feature_cols = FEATURE_COLS_PC5
    elif rat == "dsrc":
        feature_cols = FEATURE_COLS_DSRC
    else:
        raise ValueError("Invalid RAT type. Must be 'pc5' or 'dsrc'.")


    # Compute PDR and add it to dataframe
    print("Time to compute PDR...")
    compute_pdr_rolling(df, time_column, window_size_sec, packet_interval_ms)

    # Select relevant features and targets (e.g., get rid of seqnum and timestamp column)
    df = df[feature_cols].dropna()  # Ensure no NaNs

    # Normalize features and target values
    gps_scaler = normalizer_init().fit([[MIN_LAT, MIN_LON], [MAX_LAT, MAX_LON]])
    latency_scaler = normalizer_init().fit([[MIN_LATENCY], [MAX_LATENCY]])
    throughput_scaler = normalizer_init().fit([[MIN_THROUGHPUT], [MAX_THROUGHPUT]])
    pdr_scaler = normalizer_init().fit([[MIN_PDR], [MAX_PDR]])
    #sinr_scaler = normalizer_init() # will be scaled on the fly
    rsrp_scaler = normalizer_init().fit([[MIN_RSRP], [MAX_RSRP]])

    scalers = {
        'tx_latitude': gps_scaler,
        'tx_longitude': gps_scaler,
        'latency_ms': latency_scaler,
        'throughput': throughput_scaler,
        'pdr': pdr_scaler,
        'rsrp_1': rsrp_scaler,
        'rsrp_2': rsrp_scaler
    }

    print("Time to normalize data...")
    for col in feature_cols: #+ target_cols:
        if col == 'tx_latitude' or col == 'tx_longitude':
            if not gps_flag:
                df[['tx_latitude', 'tx_longitude']] = gps_scaler.transform(df[['tx_latitude', 'tx_longitude']])
                gps_flag = True
            else:
                continue
        else:
            df[col] = scalers[col].transform(df[[col]])  # Apply fitted scaler

    # Split into training/validation & new data sets
    print("Splitting data between training and future input...")
    split_idx = int(len(df) * train_ratio)
    train_val_data = df.iloc[:split_idx].copy()
    end_idx = split_idx * 2 # TODO THIS IS FOR TESTING PURPOSES, keeping the full dataset explodes RAM usage
    #new_data = df.iloc[split_idx:].copy() # TODO THIS IS FOR TESTING PURPOSES, keeping the full dataset explodes RAM usage
    new_data = df.iloc[split_idx:end_idx].copy()

    # Convert to sequences for LSTM
    print("Time to convert into 3D data...")
    print(f"___ Training data... length {len(train_val_data) - seq_length}")
    x_sequences_train, y_sequences_train = [], []
    for i in tqdm(range(len(train_val_data) - seq_length), desc="Converting training data", unit="seq"):
        x_sequences_train.append(train_val_data[feature_cols].iloc[i : i + seq_length].values)
        y_sequences_train.append(train_val_data[target_cols].iloc[i + seq_length].values)
    print(f"___ Future data... length {len(new_data) - seq_length}")
    x_sequences_new, y_sequences_new = [], []
    for i in tqdm(range(len(new_data) - seq_length), desc="Converting future data", unit="seq"):
        x_sequences_new.append(new_data[feature_cols].iloc[i : i + seq_length].values)
        y_sequences_new.append(new_data[target_cols].iloc[i + seq_length].values)

    print("Time to convert into numpy arrays...")
    return np.array(x_sequences_train), np.array(y_sequences_train), np.array(x_sequences_new), np.array(y_sequences_new), scalers

#%%
# Grab new data for predicting and re-training
def generate_new_measurement(x_new_data, y_new_data, index):
# Ensure index is within bounds
    if index >= len(x_new_data):
        raise IndexError("No more new measurements available.")
    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, 10, 6), y_new.reshape(1, *y_new.shape)
#%%
### Define LSTM model and re-training
class DataStreamGenerator(keras.utils.Sequence): # not used yet
    def __init__(self, data, labels, batch_size=1):
        self.data = data
        self.labels = np.array(labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        batch_X = self.data[start_idx:end_idx]
        batch_y = self.labels[start_idx:end_idx]
        return np.array(batch_X), np.array(batch_y)

# Define LSTM model
def build_lstm_model(timesteps, features):
    input_shape = (timesteps, features)  # Replace with the number of timesteps, and feature count
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(2)  # Output is a double target value (e.g., throughput and PDR)
    ])
    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[rmse])
    return model

# Train the model
def incremental_train(model, new_X, new_y, epochs=1, batch_size=1):
    print("Retraining with new data...")
    generator = DataStreamGenerator(new_X, new_y, batch_size=batch_size)
    model.fit(generator, epochs=epochs, verbose=1, callbacks=[EarlyStopping(patience=2)])
    #model.fit(new_X, new_y, epochs=epochs, batch_size=batch_size, verbose=0)

# Online prediction and update workflow
def predict_and_retrain(model, x_data, y_data, steps, start=0): # TODO prevent out of bounds
    for step in range(start,steps):  # Feed new data into model from new set
        # Predict
        new_x, new_y = generate_new_measurement(x_data,y_data,step)
        prediction = model.predict(new_x)
        print(prediction[0])
        print(new_y[0])
        print(f"Step {step + 1}: Predicted: {prediction[0,0]:.4f},{prediction[0,1]:.4f}, Actual: {new_y[0,0]:.4f},{new_y[0,1]:.4f}")

        # Retrain with new measurement
        print(f"Retraining model with new data at step {step}...")
        incremental_train(model, new_x, new_y)

    print("Updated model ready for future predictions.")


#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Model Training and Prediction")
    parser.add_argument('--logfile', type=str, required=True, help="Path to the log file")
    parser.add_argument('--rat', type=str, required=True, choices=['pc5', 'dsrc'], help="RAT type (pc5 or dsrc)")
    parser.add_argument("--model", type=str, help="Path to load the model, or empty for most recent, or 'none' to train a new one")
    args = parser.parse_args()

    # Set constants
    LOGFILE_PATH = args.logfile
    RAT = args.rat
    if RAT == "pc5":
        FEATURES = 4
    elif RAT == "dsrc":
        FEATURES = 6
    else:
        raise ValueError("Invalid RAT type. Must be 'pc5' or 'dsrc'.")
    if not args.model:
        # Find most recent model
        MODEL_PATH = get_latest_model(RAT)
        LOAD = True
    elif args.model == "none" :
        MODEL_PATH = None
        LOAD = False
    else:
        MODEL_PATH = args.model
        LOAD = True


    # Import data from log file
    df = pd.read_csv(LOGFILE_PATH)
    # Fill NaN values in latitude/longitude (interpolate or forward-fill)
    df["tx_latitude"] = df["tx_latitude"].interpolate().bfill()
    df["tx_longitude"] = df["tx_longitude"].interpolate().bfill()

    (X_train,y_train,X_new_data,y_new_data,scalers) = preprocess_lstm_input(df, rat=RAT, target_cols=TARGET_COLS, time_column="tx_timestamp_ms", window_size_sec=PDR_WINDOW, packet_interval_ms=TX_INTERVAL_MS, seq_length=TIMESTEPS)
    print("Preprocessing complete.")

    # Load existing model or train new one
    if LOAD and os.path.exists(MODEL_PATH):
        print("Loading existing model: " + MODEL_PATH)
        model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})
        if model:
            print("Model loaded successfully: " + MODEL_PATH)
        else:
            raise ValueError("Failed to load model.")
    else:
        if not LOAD:
            print("No model selected. Building...")
        elif not os.path.exists(MODEL_PATH):
            print("No model found. Building...")
        model = build_lstm_model(TIMESTEPS, FEATURES)
        # model.summary() # Print model summary
        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True) # Stop training if loss does not improve
        print("Initial training...")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
        lstm_path = "model_" + RAT + "_" + str(int(time.time())) + ".keras"
        model.save(lstm_path)
        print(f"Model saved to {lstm_path}")

    # Predict and retrain
    position = 0
    while True:
        print("____________________________________________________")
        user_input = input("Prediction and retraining. Enter 'steps' to predict and 'starting' point in the new values dataset, separated by a space (or 'exit' to quit, or leave empty for one step): ")
        if user_input.lower() == 'exit':
            break
        elif not user_input:
            predict_and_retrain(model, X_new_data, y_new_data, position+1, position)
            position += 1
        else:
            try:
                steps, start = map(int, user_input.split())
                predict_and_retrain(model, X_new_data, y_new_data, steps+start, start)
                position = start + steps
                lstm_path = "model_" + RAT + "_" + str(int(time.time())) + ".keras"
                model.save(lstm_path)
                print(f"Model saved to {lstm_path}")
            except ValueError:
                print("Invalid input. Please enter two integers separated by a space.")
#%%
