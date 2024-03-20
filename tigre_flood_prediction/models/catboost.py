import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
import pickle5

def clean_and_encode_data_cb():
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

def create_model(data):
    # Define feature matrix X and target vector y
    X = data.drop(columns=['date', 'time', 'alerta_crecida'])
    y = data['alerta_crecida']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the CatBoost classifier
    model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, random_state=42)
    model.fit(X_train, y_train, verbose=False)

    # Make predictions on the test set
    y_pred_catboost = model.predict(X_test)

    # Visualize the classification report
    print("Classification Report - CatBoost Classifier:")
    print(classification_report(y_test, y_pred_catboost))

    return model

def main_cb():
    data = clean_and_encode_data_cb()
    model = create_model(data)

    with open('tigre_flood_prediction/models/cat_boost_model.pkl','wb') as f:
        pickle5.dump(model, f)

if __name__ == '__main__':
    main_lr()
