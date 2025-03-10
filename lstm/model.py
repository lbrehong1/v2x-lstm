import numpy as np
import keras
import tensorflow.keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt

#%%
### Define loss function
def rmse(y_true, y_pred):
    y_true = K.cast(y_true, np.float32) # Ensure same type because auto-casting pulls float64
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

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

# Define GRU model
def build_gru_model(timesteps, features):
    input_shape = (timesteps, features)  # Replace with the number of timesteps, and feature count
    model = Sequential([
        GRU(64, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(2)  # Output is a double target value (e.g., throughput and PDR)
    ])
    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[rmse])
    return model

# Define RNN model
def build_rnn_model(timesteps, features):
    input_shape = (timesteps, features)  # Replace with the number of timesteps, and feature count
    model = Sequential([
        SimpleRNN(64, activation='tanh', return_sequences=False, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(2)  # Output is a double target value (e.g., throughput and PDR)
    ])
    model.compile(Adam(learning_rate=0.001), loss='mse', metrics=[rmse])
    return model

#%%
### Define data stream generator
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

#%%
### Train the model
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
        print(f"___ Retraining model with new data at step {step + 1}...")
        incremental_train(model, new_x, new_y)

    print("___ Updated model ready for future predictions.")

# Grab new data for predicting and re-training
def generate_new_measurement(x_new_data, y_new_data, index):
    # Ensure index is within bounds
    if index >= len(x_new_data):
        raise IndexError("!!! No more new measurements available.")

    x_new = x_new_data[index]
    y_new = y_new_data[index]
    return x_new.reshape(1, *x_new.shape), y_new.reshape(1, *y_new.shape)

#%%
def automatic_train(model, X_new_data, y_new_data, batch_size=32, N=500, validation=0.15, log_file="prediction_log.csv"):
    x_new, y_new = [], []
    log_entries = []
    validation_split = int(validation * len(X_new_data))  # 15% for validation

    X_val, y_val = X_new_data[:validation_split], y_new_data[:validation_split]
    X_new_data, y_new_data = X_new_data[validation_split:], y_new_data[validation_split:]

    for i in tqdm(range(len(X_new_data)), desc="Processing new data", unit="seq"):
        x_new.append(X_new_data[i])
        y_new.append(y_new_data[i])


        if len(x_new) >= N:  # Retrain every 500 new points
            # Log predictions before training
            for j in range(len(x_new)):
                #print(x_new[j][-1])
                lat, lon = x_new[j][-1][:2]  # Extract last lat, lon in sequence
                pred = model.predict(np.expand_dims(x_new[j], axis=0))[0]
                actual = y_new[j]
                error = np.sqrt(np.mean((pred - actual) ** 2))  # RMSE
                log_entries.append(f"{lat},{lon},{pred[0]},{actual[0]},{error}")

            # Train the model
            generator = DataStreamGenerator(x_new, y_new, batch_size)
            model.fit(generator, epochs=1, verbose=1, callbacks=[EarlyStopping(patience=2)], validation_data=(np.array(X_val), np.array(y_val)))

            x_new, y_new = [], []  # Clear batch

        with open(log_file, "a") as f:
            for entry in log_entries:
                f.write(entry + "\n")


#%%
def plot_losses(history):
    # Plot Training & Validation Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Training vs Validation Loss')
    plt.show()