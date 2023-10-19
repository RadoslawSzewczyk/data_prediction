import tensorflow as tf
import numpy as np
import pandas as pd
import os

def get_small_data(start, stop):
    #10m records
    file = open("/home/radek/fill_all_gaps/Indirect_Trade_2016.cma", 'r', encoding='utf-8-sig')
    save = open("testOutput.csv", 'a')
    counter = 0
    for i in file:
        counter+=1
        if counter < start:
            continue
        line = i.strip().replace("\"", "")
        lineNoUSD = line.replace("USD", "1")
        lineNoKG = lineNoUSD.replace("kg", "0")
        lineNoTrade = lineNoKG.replace("IISI:Indirect_Trade,", "")

        save.write(lineNoTrade)
        save.write("\n")
        if counter == stop:
            break


def create_model():
    fill_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    fill_model.compile(loss='mean_squared_error', optimizer='adam')
    return fill_model


def main():
    get_small_data(0, 10000)
    output = open("testOutput.csv",'r', encoding='utf-8-sig')
    train_data = pd.read_csv(output, names=["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"],on_bad_lines='skip')

    fill_features = train_data.copy()
    fill_labels = fill_features.pop("Volume")

    fill_features = fill_features.values
    fill_labels = fill_labels.values

    fill_features = fill_features.astype(float)
    fill_labels = fill_labels.astype(float)

    fill_model = create_model()
    fill_model.summary()

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1)

    fill_model.load_weights(checkpoint_path)

    #fill_model.fit(fill_features, fill_labels, epochs=10, callbacks=[cp_callback])

    # new_data = np.array([["feature1", "feature2", "feature3", "feature4", "feature5", "feature6"]])
    # predictions = fill_model.predict(new_data)
    os.remove("/home/radek/fill_all_gaps/testOutput.csv")

main()