import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle5

import sys 
sys.path.append('tigre_flood_prediction/models')

def clean_and_encode_data_lr():
    
    data = pd.read_csv('/Users/axelcasas/Documents/1_Projects/2-data-science/tigre_flood_prediction/data/featurized/tigre_dataset.csv')
    data = data.drop(columns=['Unnamed: 0', 'visibility', 'time_crecida', 'degree'])
        
    # One-hot encode 'weather' column
    one_hot_encoded_weather = pd.get_dummies(data['weather'])
    data = pd.concat([data, one_hot_encoded_weather], axis=1)
    data.drop(columns=['weather'], inplace=True)
    
    # One-hot encode 'wind_direction' column
    one_hot_encoded_wind = pd.get_dummies(data['wind_direction'], prefix='wind')
    data = pd.concat([data, one_hot_encoded_wind], axis=1)
    data.drop(columns=['wind_direction'], inplace=True)
    data = data.dropna(subset=['heigh_m'])
    
    return data

def create_logistic_regression_model(data):

    X = data.drop(columns=['date', 'time', 'alerta_crecida'])
    y = data['alerta_crecida']

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Initialize Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model
    train_accuracy = accuracy_score(y_train, y_pred_train).round(2)
    test_accuracy = accuracy_score(y_test, y_pred_test).round(2)
    classification_rep = classification_report(y_test, y_pred_test)

    print("Train Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)
    print('Classification report:\n', classification_rep)

    return model, scaler

def main_lr():
    data = clean_and_encode_data_lr()
    model, scaler = create_logistic_regression_model(data)

    with open('tigre_flood_prediction/models/logistic_regression_model.pkl','wb') as f:
        pickle5.dump(model, f)
    
    with open('tigre_flood_prediction/models/logistic_regression_scaler.pkl','wb') as f:
        pickle5.dump(scaler, f)

if __name__ == '__main__':
    main_lr()
