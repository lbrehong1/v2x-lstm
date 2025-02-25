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
from dotenv import load_dotenv
load_dotenv()

import time
import argparse
import glob
import re
import pandas as pd
#import matplotlib.pyplot as plt
import os
import multiprocessing
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from keras.callbacks import EarlyStopping
from keras.models import load_model
from data_preprocessing import preprocess_lstm_input
from model import build_lstm_model, predict_and_retrain, rmse

#%%
#LOGFILE_PATH = "~/Documents/icccn-lstm/logs_cohda/trim_dsrc.txt"
#MODEL_PATH = "lstm_model.keras"

TIMESTEPS = 10
FEATURES = 6
EPOCHS = 5
SAMPLES = 20000 # number of samples to generate when testing
TRAIN_RATIO = 0.4 # ratio of samples to use for training
BATCH_SIZE = 32 # batch size for training, e.g. what is the number of samples to use for each epoch
VALIDATION_SPLIT = 0.15
TX_INTERVAL_MS = 20 # in ms, 50 packets per second
#TX_INTERVAL_MS = 100 # in ms, 10 packets per second
PDR_WINDOW = 1 # in seconds
TARGET_COLS = ['latency_ms', 'pdr']
MODEL_DIR = "models"


#%%
# Ensure the models directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_latest_model(rat):
    """
    Finds the most recent model file based on the RAT type.
    :param rat: RAT type ('pc5' or 'dsrc').
    :return: Path to the most recent model file.
    """
    model_files = glob.glob(os.path.join(MODEL_DIR, "model_" + rat + "_" + "*.keras"))
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

# # data normalizer
# def normalizer_init():
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     return scaler
#
# # Normalize data. Requires an array of single values, except for GPS that requires [latitude, longitude] elements
# def normalize(scaler, data):
#     return scaler.transform(data)


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
        MODEL_PATH = os.path.join(MODEL_DIR, get_latest_model(RAT))
        LOAD = True
    elif args.model == "none" :
        MODEL_PATH = None
        LOAD = False
    else:
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.model))
        LOAD = True


    # Import data from log file
    df = pd.read_csv(LOGFILE_PATH)
    # Fill NaN values in latitude/longitude (interpolate or forward-fill)
    df["tx_latitude"] = df["tx_latitude"].interpolate().bfill()
    df["tx_longitude"] = df["tx_longitude"].interpolate().bfill()

    (X_train,y_train,X_new_data,y_new_data,scalers) = preprocess_lstm_input(df, rat=RAT, target_cols=TARGET_COLS, time_column="tx_timestamp_ms", window_size_sec=PDR_WINDOW, packet_interval_ms=TX_INTERVAL_MS, seq_length=TIMESTEPS, train_ratio=TRAIN_RATIO)
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
        lstm_path = os.path.join(MODEL_DIR, "model_" + RAT + "_" + str(int(time.time())) + ".keras")
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
                lstm_path = os.path.join(MODEL_DIR, "model_" + RAT + "_" + str(int(time.time())) + ".keras")
                model.save(lstm_path)
                print(f"Model saved to {lstm_path}")
            except ValueError:
                print("Invalid input. Please enter two integers separated by a space.")
#%%
