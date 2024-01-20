import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def get_small_data(start, stop):
    # Process a chunk of data from a large file
    processed_lines = []
    with open("Indirect_Trade_2016.cma", 'r', encoding='utf-8-sig') as file:
        for counter, line in enumerate(file, 1):
            if counter > stop:
                break
            if counter >= start:
                processed_line = line.strip().replace("\"", "").replace("USD", "1").replace("kg", "0").replace("IISI:Indirect_Trade,", "")
                processed_lines.append(processed_line.split(',')) 

    train_data = pd.DataFrame(processed_lines, columns=["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"])
    return train_data

def get_test_data(file_path, start, stop):

    processed_lines = []
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        for counter, line in enumerate(file, 1):
            if counter > stop:
                break
            if counter >= start:
                processed_line = line.strip().replace("\"", "").replace("USD", "1").replace("kg", "0").replace("IISI:Indirect_Trade,", "")
                processed_lines.append(processed_line.split(','))

    test_data = pd.DataFrame(processed_lines, columns=["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"])
    return test_data

def prepare_test_data(file_path, start, stop):

    test_data = get_test_data(file_path, start, stop)

    # Convert categorical columns to numeric
    label_encoders = {}
    categorical_columns = ['Product', 'Reporter', 'Partner', 'Year', 'Flow', 'Unit']  # Update as per your data
    for col in categorical_columns:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col].astype(str))
        label_encoders[col] = le

    # Separate features and labels
    test_labels = test_data.pop("Volume").astype(np.float32)
    test_features = test_data.astype(np.float32)

    return test_features.values, test_labels.values


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(6,)),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
    return model


def prepare_data(data_frame):
    # Convert categorical columns to numeric
    label_encoders = {}
    categorical_columns = ['Product', 'Reporter', 'Partner', 'Year', 'Flow', 'Unit']  # Update this list as per your data
    for col in categorical_columns:
        le = LabelEncoder()
        data_frame[col] = le.fit_transform(data_frame[col].astype(str))
        label_encoders[col] = le

    # Separate features and labels
    fill_labels = data_frame.pop("Volume").astype(np.float32)
    fill_features = data_frame.astype(np.float32)

    return fill_features.values, fill_labels.values


def train_model_in_batches(model, start=0, batch_size=100000, callbacks=None):
    # Train the model in batches
    end = start + batch_size
    while True:
        data_frame = get_small_data(start, end)
        if data_frame.empty:
            break
        fill_features, fill_labels = prepare_data(data_frame)
        model.fit(fill_features, fill_labels, epochs=10, batch_size=32, callbacks=callbacks)
        start += batch_size
        end += batch_size

def main():
    fill_model = create_model()

    checkpoint_path = "training/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        save_freq='epoch') 

    train_model_in_batches(fill_model, callbacks=[cp_callback])

    fill_model.save_weights(checkpoint_path.format(epoch=0))

    fill_model.summary()

def load_model_from_checkpoint(model, checkpoint_path):

    status = model.load_weights(checkpoint_path)
    status.expect_partial()
    return model

def evaluate_model(model, test_data):

    test_features, test_labels = test_data
    loss = model.evaluate(test_features, test_labels)
    print(f"Test Loss: {loss}")

def test():
    fill_model = create_model()
    file_path = "Indirect_Trade_2017.cma"

    checkpoint_path = "training/cp-0010.ckpt"

    fill_model = load_model_from_checkpoint(fill_model, checkpoint_path)

    test_features, test_labels = prepare_test_data(file_path, start=100001, stop=200000)  # Adjust the range as needed

    evaluate_model(fill_model, (test_features, test_labels))

if __name__ == "__main__":
    main()
    test()