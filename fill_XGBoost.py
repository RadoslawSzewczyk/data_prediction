import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import joblib
import os

def safe_load_data(filepath, start, stop):
    try:
        # Process a chunk of data from a large file
        processed_lines = []
        with open(filepath, 'r', encoding='utf-8-sig') as file:
            for counter, line in enumerate(file, 1):
                if counter > stop:
                    break
                if counter >= start:
                    processed_line = line.strip().replace("\"", "").replace("USD", "1").replace("kg", "0").replace("IISI:Indirect_Trade,", "")
                    processed_lines.append(processed_line.split(',')) 

        return pd.DataFrame(processed_lines, columns=["Product", "Reporter", "Partner", "Year", "Flow", "Unit", "Volume"])
    except FileNotFoundError:
        print(f"File {filepath} not found. Please check the file path.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def prepare_data_for_prediction(data_frame, label_encoders):
    # Drop columns that are not features
    data_frame = data_frame.drop(columns=['Year', 'Flow', 'Unit', 'Volume'], errors='ignore')

    for col, le in label_encoders.items():
        if col in data_frame:
            # Transform and update the column in place
            # Handle unseen labels by checking if they are in the training set
            mask = data_frame[col].isin(le.classes_)
            if not mask.all():
                # Handle unseen labels by assigning a default value (e.g., 'Unknown')
                data_frame.loc[~mask, col] = 'Unknown'
                le.classes_ = np.append(le.classes_, 'Unknown')
            data_frame[col] = le.transform(data_frame[col])

    # Convert all columns to numeric
    data_frame = data_frame.apply(pd.to_numeric, errors='coerce')

    return data_frame

def load_xgboost_model(filename):
    model = xgb.XGBRegressor()
    model.load_model(filename)
    return model

def prepare_new_data(new_data, label_encoders):
    for col in new_data.columns:
        if col in label_encoders:
            new_data[col] = label_encoders[col].transform(new_data[col].astype(str))
        else:
            new_data[col] = pd.to_numeric(new_data[col], errors='coerce')

    return new_data


# Function to make predictions with the model
def make_prediction(model, new_data, label_encoders):
    new_features = prepare_new_data(new_data, label_encoders)
    predictions = model.predict(new_features)
    # If your target variable was logged during training, apply the inverse transformation
    #predictions = np.expm1(predictions)
    return predictions


def prepare_data(data_frame):
    data_frame = data_frame.drop(columns=['Year', 'Flow', 'Unit'], errors='ignore')

    # Label encoding for categorical features
    label_encoders = {}
    categorical_columns = ['Product', 'Reporter', 'Partner']

    for col in categorical_columns:
        le = LabelEncoder()
        data_frame[col] = le.fit_transform(data_frame[col].astype(str))
        label_encoders[col] = le

    # Convert 'Volume' to numeric and apply log transformation
    data_frame['Volume'] = pd.to_numeric(data_frame['Volume'], errors='coerce')
    data_frame['Volume'] = np.log1p(data_frame['Volume'])

    fill_labels = data_frame.pop("Volume").astype(float)
    fill_features = data_frame

    return fill_features, fill_labels, label_encoders

def train_xgboost_model(start, stop):
    data_frame = safe_load_data("Indirect_Trade_2016.cma", start, stop)
    if data_frame is None:  # Check if data loading was successful
        return
    fill_features, fill_labels, label_encoders = prepare_data(data_frame)  # Capture all three items


    X_train, X_test, y_train, y_test = train_test_split(fill_features, fill_labels, test_size=0.2, random_state=42)

    # Initialize XGBoost model with adjusted parameters
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.5,
        learning_rate=0.05,
        max_depth=7,
        alpha=10,
        n_estimators=200
    )
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    joblib.dump(label_encoders, 'label_encoders.pkl')
    # Calculate MSE using the actual values of y_test
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-Validated MSE: {-scores.mean()}")

    # Plotting Feature Importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.show()

    # Plotting actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs. Predicted Values')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
    plt.show()

def makeModel():
    # Define start and stop for reading data, if your data is too large, you might need to read it in chunks
    start, stop = 0, 1000000  # Adjust as needed
    
    # Read the data
    data_frame = safe_load_data("Indirect_Trade_2016.cma", start, stop)
    
    # Prepare the data
    fill_features, fill_labels, label_encoders = prepare_data(data_frame)
    
    # Split the data into training and testing set
    X_train, X_test, y_train, y_test = train_test_split(fill_features, fill_labels, test_size=0.2, random_state=42)
    
    # Initialize the XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        colsample_bytree=0.5,
        learning_rate=0.05,
        max_depth=7,
        alpha=10,
        n_estimators=200
    )
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Save the model
    model.save_model('xgb_model.json')
    
    # Save the label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    
    # If log transformation was used during training, reverse it before calculating MSE
    mse = mean_squared_error(np.expm1(y_test), np.expm1(y_pred)) if 'Volume' in data_frame else mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(np.expm1(y_test), np.expm1(y_pred), alpha=0.3) if 'Volume' in data_frame else plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.plot([min(np.expm1(y_test)), max(np.expm1(y_test))], [min(np.expm1(y_test)), max(np.expm1(y_test))], 'k--', lw=4) if 'Volume' in data_frame else plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    plt.show()

def plot_residuals(actual, predicted):
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    residuals = actual - predicted
    if actual.shape[0] != predicted.shape[0]:
        raise ValueError(f"The length of actual ({actual.shape[0]}) and predicted ({predicted.shape[0]}) do not match.")
    

    plt.figure(figsize=(10, 6))
    plt.scatter(predicted, residuals, alpha=0.3)
    plt.hlines(y=0, xmin=predicted.min(), xmax=predicted.max(), colors='red', linestyles='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()


def main():
    # Load the trained model and label encoders
    trained_model = load_xgboost_model('xgb_model.json')
    label_encoders = joblib.load('label_encoders.pkl')

    # Read the entire file for prediction
    prediction_data = safe_load_data("Indirect_Trade_2016.cma", 0, float('inf'))

    if 'Volume' in prediction_data:
        actual_values = pd.to_numeric(prediction_data['Volume'], errors='coerce')

    prediction_features = prepare_data_for_prediction(prediction_data, label_encoders)

    # Make predictions for all entries
    predictions = trained_model.predict(prediction_features)
 
    # If you want to store the predictions alongside the features
    prediction_data['Predicted_Volume'] = predictions
    
    #prediction_data.to_csv('predicted_volumes.csv', index=False)
    plot_residuals(actual_values, prediction_data)

if __name__ == "__main__":
    if os.path.exists('xgb_model.json') and os.path.exists('label_encoders.pkl'):
        main()
    else:
        print("Model or label encoders not found. Training new model.")
        makeModel()
