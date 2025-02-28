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
from model import build_lstm_model, build_gru_model, build_rnn_model, predict_and_retrain, rmse

#%%
TIMESTEPS = 10
FEATURES = 6
EPOCHS = 20
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

def get_latest_model(model_type, rat):
    """
    Finds the most recent model file based on the RAT type.
    :param type: Model type ('lstm', 'gru', 'rnn').
    :param rat: RAT type ('pc5' or 'dsrc').
    :return: Path to the most recent model file.
    """
    model_files = glob.glob(os.path.join(MODEL_DIR, model_type + "_" + rat + "_" + "*.keras"))
    model_files_with_time = []
    regex = re.compile(rf"{model_type}_{rat}_(\d+)\.keras")
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
    parser.add_argument('--data', type=str, required=True, help="Path to the log file")
    parser.add_argument('--rat', type=str, required=True, choices=['5g', 'pc5', 'dsrc'], help="RAT type (5g, pc5 or dsrc)")
    parser.add_argument("--load", type=str, help="Path to load the model, or empty for most recent, or 'none' to train a new one")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train the model")
    parser.add_argument("--model", type=str, required=True, choices=['lstm', 'gru', 'rnn'], help="Type of model to use (lstm, gru, rnn.)")
    args = parser.parse_args()

    # Set constants
    LOGFILE_PATH = args.data
    RAT = args.rat
    MODEL_TYPE = args.model
    if args.epochs:
        EPOCHS = int(args.epochs)

    if RAT == "pc5":
        FEATURES = 4
    elif RAT == "dsrc":
        FEATURES = 6
    elif RAT == "5g":
        FEATURES = 6
    else:
        raise ValueError("Invalid RAT type. Must be '5g', 'pc5' or 'dsrc'.")

    if args.load == "none":
        MODEL_PATH = None
        LOAD = False
    elif args.load:
        MODEL_PATH = os.path.join(MODEL_DIR, os.path.basename(args.load))
        LOAD = True
    else:
        part = get_latest_model(MODEL_TYPE, RAT)
        if part:
            MODEL_PATH = part
            LOAD = True
        else:
            MODEL_PATH = None
            LOAD = False

    # Import data from log file
    if not os.path.exists(LOGFILE_PATH):
        raise ValueError(f"!!! Log file not found. {LOGFILE_PATH}")
    elif not LOGFILE_PATH.__contains__(RAT):
        raise ValueError(f"!!! Log file and argument RAT type do not match. {LOGFILE_PATH}")
    df = pd.read_csv(LOGFILE_PATH)
    # Fill NaN values in latitude/longitude (interpolate or forward-fill)
    df["tx_latitude"] = df["tx_latitude"].interpolate().bfill()
    df["tx_longitude"] = df["tx_longitude"].interpolate().bfill()


    # Load existing model or train new one
    # Check for consistency of arguments
    if LOAD and os.path.exists(MODEL_PATH):
        if not MODEL_PATH.__contains__(RAT):
            raise ValueError("!!! Model and log file RAT types do not match.")
        if not MODEL_PATH.__contains__(MODEL_TYPE):
            raise ValueError("!!! Model file and model type do not match.")

        # Load existing model
        print("___ Loading existing model: " + MODEL_PATH)
        model = load_model(MODEL_PATH, custom_objects={'rmse': rmse})
        if model and MODEL_PATH.__contains__(RAT):
            print("___ Model loaded successfully: " + MODEL_PATH)
        else:
            raise ValueError("!!! Failed to load model")#. Does the model's RAT type match the argument RAT type?")
        print("___ Starting data preprocessing.")
        (X_train,y_train,X_new_data,y_new_data,scalers) = preprocess_lstm_input(df, new=False, rat=RAT, target_cols=TARGET_COLS, time_column="tx_timestamp_ms", window_size_sec=PDR_WINDOW, packet_interval_ms=TX_INTERVAL_MS, seq_length=TIMESTEPS, train_ratio=TRAIN_RATIO)
        print("___ Preprocessing complete.")
    else:
        # Train new model
        if not LOAD:
            print("___ No model selected. Building...")
        elif not os.path.exists(MODEL_PATH):
            print("___ No model found. Building...")
        print("___ Starting data preprocessing.")
        (X_train,y_train,X_new_data,y_new_data,scalers) = preprocess_lstm_input(df, new=True, rat=RAT, target_cols=TARGET_COLS, time_column="tx_timestamp_ms", window_size_sec=PDR_WINDOW, packet_interval_ms=TX_INTERVAL_MS, seq_length=TIMESTEPS, train_ratio=TRAIN_RATIO)
        print("___ Preprocessing complete.")

        if MODEL_TYPE == "lstm":
            model = build_lstm_model(TIMESTEPS, FEATURES)
        elif MODEL_TYPE == "gru":
            model = build_gru_model(TIMESTEPS, FEATURES)
        elif MODEL_TYPE == "rnn":
            model = build_rnn_model(TIMESTEPS, FEATURES)
        else:
            raise ValueError("!!! Invalid model type. Must be 'lstm', 'gru' or 'rnn'.")
        # model.summary() # Print model summary
        early_stopping = EarlyStopping(monitor='loss', patience=5,
                                       restore_best_weights=True)  # Stop training if loss does not improve
        print("___ Initial training...")
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopping], validation_split=VALIDATION_SPLIT, verbose=1)
        save_path = os.path.join(MODEL_DIR, MODEL_TYPE + "_" + RAT + "_" + str(int(time.time())) + ".keras")
        model.save(save_path)
        print(f"___ Model saved to {save_path}")

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
                save_path = os.path.join(MODEL_DIR, MODEL_TYPE + "_" + RAT + "_" + str(int(time.time())) + ".keras")
                model.save(save_path)
                print(f"Model saved to {save_path}")
            except ValueError:
                print("Invalid input. Please enter two integers separated by a space.")
#%%
