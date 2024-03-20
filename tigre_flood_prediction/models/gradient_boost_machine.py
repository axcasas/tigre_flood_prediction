import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle5

def clean_and_encode_data_gb():
    
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

def create_gradient_boost_machine_model(data):

    # Select variables
    X = data.drop(columns=['date', 'time', 'alerta_crecida'])
    y = data['alerta_crecida']

    # Scale data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Gradient Boosting classifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)

    # Train the Gradient Boosting classifier
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_gb = model.predict(X_test)

    # Calculate accuracy
    accuracy_gb = accuracy_score(y_test, y_pred_gb)
    print("Gradient Boosting Classifier Accuracy:", accuracy_gb)

    # Print classification report
    print("Classification Report - Gradient Boosting Classifier:")
    print(classification_report(y_test, y_pred_gb))

    return model, scaler

def main_gb():
    data = clean_and_encode_data_gb()
    model, scaler = create_gradient_boost_machine_model(data)

    with open('tigre_flood_prediction/models/gradient_boost_machine_model.pkl','wb') as f:
        pickle5.dump(model, f)
    
    with open('tigre_flood_prediction/models/gradient_boost_machine_scaler.pkl','wb') as f:
        pickle5.dump(scaler, f)

if __name__ == '__main__':
    main_gb()