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
        if(counter == 10000000):
            return train_data.clear()
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
    file.close()
    save.close()

    output = open("testOutput.csv",'r', encoding='utf-8-sig')
    train_data = pd.read_csv(output, names=["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"],on_bad_lines='skip')
    output.close()
    os.remove("/home/radek/fill_all_gaps/testOutput.csv")

    return train_data


def create_model():
    fill_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    fill_model.compile(loss='mean_squared_error', optimizer='adam')
    return fill_model

def prepare_data(start, stop):
    fill_features =get_small_data(start, stop)
    fill_labels = fill_features.pop("Volume")
 
    fill_features = fill_features.values
    fill_labels = fill_labels.values

    # fill_features = fill_features.astype(float)
    # fill_labels = fill_labels.astype(float)
    # print(fill_features)
    # print(fill_labels)
    pair = (fill_features, fill_labels)
    
    return pair

def feed(fill_model):
    x = 0
    y = 100000
    while():
        if not prepare_data(x, y):
            break
        dataPair = prepare_data(x, y)
        fill_model.fit(dataPair[0],dataPair[1], epochs=10)
        x += 100000
        y += 100000


def main():

    fill_model = create_model()

    feed(fill_model)
    #dataTest = prepare_data(0, 10000)
    #print(dataPair[0])

    #print(fill_model.evaluate(dataTest[0], dataTest[1]))

    new_data = np.array([["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"]])
    predictions = fill_model.predict(new_data)
    fill_model.summary()

main()