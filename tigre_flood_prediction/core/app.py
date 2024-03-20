import sys 
sys.path.append('tigre_flood_prediction/models')

import streamlit as st 
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle5 as pickle
from logistic_regression import clean_and_encode_data_lr, create_logistic_regression_model, main_lr
from random_forest import clean_and_encode_data_rf, create_random_forest_model, main_rf
from gradient_boost_machine import clean_and_encode_data_gb, main_gb

def sidebar():
    st.sidebar.image("/Users/axelcasas/Documents/1_Projects/2-data-science/tigre_flood_prediction/tigre_flood_prediction/utils/tigre_municipio.png")
    st.sidebar.header('Predicción de Alertas Crecida Tigre')
    data = clean_and_encode_data_lr()

    # model_choice = st.sidebar.radio("Select Model", ("Logistic Regression", "Random Forest"))

    labels = [
        ("Temperature (Celsius)", "temperature_celsius"),
        ("Humidity", "humidity"),
        ("Barometer (mbar)", "barometer_mbar"),
        ("Heigh (m)", "heigh_m")
    ]

    input_data = {}

    for label, key in labels:
        input_data[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )

    # Add checkboxes for weather conditions
    st.sidebar.subheader("Weather Conditions")
    for condition in ['Clear', 'Cloudy', 'Fog', 'Light Rain', 'Rainy', 'Thunderstorms']:
        input_data[condition] = st.sidebar.checkbox(condition)

    # Add checkboxes for wind directions if they exist in the dataset
    wind_directions = [col for col in data.columns if col.startswith('wind_')]
    if wind_directions:
        st.sidebar.subheader("Wind Direction")
        for direction in wind_directions:
            input_data[direction] = st.sidebar.checkbox(direction)

    return input_data

def radar_chart(input_data):

    def get_scaled_data(input):
        
        data = clean_and_encode_data_lr() 

        X = data[['temperature_celsius', 'humidity', 'barometer_mbar', 'heigh_m']]  # Only selecting specific features

        scaled_dict = {}

        for key, value in input.items():
            if key in X.columns:
                max_value = X[key].max()
                min_value = X[key].min()
                scaled_value = (value - min_value) / (max_value - min_value)
                scaled_dict[key] = scaled_value

        return scaled_dict

    categories = [
        'Temperature (Celsius)', 'Humidity', 'Barometer (mbar)', 'Heigh (m)'
    ]

    # Scale the input data
    scaled_input_data = get_scaled_data(input_data)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            scaled_input_data.get('temperature_celsius', 0),  # Default value if key is missing
            scaled_input_data.get('humidity', 0),
            scaled_input_data.get('barometer_mbar', 0),
            scaled_input_data.get('heigh_m', 0)
        ],
        theta=categories,
        fill='toself',
        name='Weather Data'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig


def make_predictions(input_data, model):

    if model == "Logistic Regression":
        model_file = "tigre_flood_prediction/models/logistic_regression_model.pkl"
        data_func = clean_and_encode_data_lr
        pred_func = main_lr
        
    elif model == "Random Forest":
        model_file = "tigre_flood_prediction/models/random_forest_model.pkl"
        data_func = clean_and_encode_data_rf
        pred_func = main_rf

    elif model == "Gradient Boosting Machine":
        model_file = "/Users/axelcasas/Documents/1_Projects/2-data-science/tigre_flood_prediction/tigre_flood_prediction/models/gradient_boost_machine_model.pkl"
        data_func = clean_and_encode_data_gb
        pred_func = main_gb

    model = pickle.load(open(model_file, "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    prediction = model.predict(input_array)
    probabilities = model.predict_proba(input_array)

    st.subheader("Predicción de Alerta de Crecida")
    st.write("La alerta de crecida es:")
    
    if prediction[0] == 'NO':
        st.write("<span class='diagnosis benign'>NO</span>",unsafe_allow_html=True),
    else:
        st.write("<span class='diagnosis malicious'>YES</span>", unsafe_allow_html=True)
    
    st.write("Probabilidad de que sea NO: ", probabilities[0][0])
    st.write("Probabilidad de que sea YES: ", probabilities[0][1])

def main():
    
    st.set_page_config( # Step 1. Set page configuration
        page_title = 'Alerta Crecida Tigre',
        layout = "wide",
        initial_sidebar_state = "expanded"
    )

    input_data = sidebar() # Step 2. Call sidebar function

    st.sidebar.subheader("Select Model")

    model_choice = st.sidebar.radio("Select Model", ("Logistic Regression", "Random Forest", "Gradient Boosting Machine"))

    if model_choice == "Logistic Regression":
        radar_chart_data = radar_chart(input_data)  # Retrieve data for radar chart
        with st.container():
            st.title("🐯 Alerta Crecida Tigre")
            st.write("Esta es una aplicación de Machine Learning (usando un modelo de regresión logística)")
            st.write("para predecir alertas de crecida en Tigre, Buenos Aires, Argentina")

        col1, col2 = st.columns([4,1])

        with col1:
            chart = radar_chart_data
            st.plotly_chart(chart)
        with col2:
            make_predictions(input_data, model="Logistic Regression")  # Pass model type as an argument

    elif model_choice == "Random Forest":
        radar_chart_data_rf = radar_chart(input_data)  # Retrieve data for radar chart
        with st.container():
            st.title("🐯 Alerta Crecida Tigre")
            st.write("Esta es una aplicación de Machine Learning (usando un modelo de Random Forest)")
            st.write("para predecir alertas de crecida en Tigre, Buenos Aires, Argentina")

        col1, col2 = st.columns([4,1])

        with col1:
            chart = radar_chart_data_rf
            st.plotly_chart(chart)
        with col2:
            make_predictions(input_data, model="Random Forest")  # Pass model type as an argument

    elif model_choice == "Gradient Boosting Machine":
        radar_chart_data_gb = radar_chart(input_data)  # Retrieve data for radar chart
        with st.container():
            st.title("🐯 Alerta Crecida Tigre")
            st.write("Esta es una aplicación de Machine Learning (usando un modelo de Gradient Boosting Machine)")
            st.write("para predecir alertas de crecida en Tigre, Buenos Aires, Argentina")

        col1, col2 = st.columns([4,1])

        with col1:
            chart = radar_chart_data_gb
            st.plotly_chart(chart)
        with col2:
            make_predictions(input_data, model="Gradient Boosting Machine")  # Pass model type as an argument

if __name__ == "__main__":
    main()