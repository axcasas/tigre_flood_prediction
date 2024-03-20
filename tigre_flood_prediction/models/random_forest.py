import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle5

def clean_and_encode_data_rf():
    
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

def create_random_forest_model(data):
    X = data.drop(columns=['date', 'time', 'alerta_crecida'])
    y = data['alerta_crecida']

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_rf = rf_classifier.predict(X_test)

    # Calculate accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Random Forest Classifier Accuracy:", accuracy_rf)

    # Generate classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf))

    # Generate confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_rf))

    return rf_classifier, scaler

def main_rf():
    data = clean_and_encode_data_rf()
    model, scaler = create_random_forest_model(data)

    with open('tigre_flood_prediction/models/random_forest_model.pkl','wb') as f:
        pickle5.dump(model, f)
    
    with open('tigre_flood_prediction/models/random_forest_scaler.pkl','wb') as f:
        pickle5.dump(scaler, f)

if __name__ == '__main__':
    main_rf()