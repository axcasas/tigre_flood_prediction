# Tigre-Floods-Prediction
=======

About This Project
==
People in areas like Tigre, the Paraná delta and more have a problem: **they are victims of frequent flooding**. 

In some periors, the level of the river is continually rising, and the flooding of the Paraná can coincide with "*sudestadas*", which directly affects the delta and leaves it trapped due to the "hydraulic plug effect" of the river. However, what affects them most is the intense private suburbanization. 

Neighbors usually find out about river flooding through social networks and WhatsApp messages. **People go to information and not information to people**. This dynamic is not efficient or effective and can leave people on the sidelines. 

Our goal is to reverse this. We seek to ensure that all people who may be potentially affected by floods can be aware of them in a timely manner, thus allowing them to plan their days accordingly.

In order to do this, we constructed a dataset with climatic, tidal, and alert variables from Tigre's government to predict using Machine Learning models when Tigre will have an alert of flood. 

Finally, we developed an application using Streamlit to study which variables are more relevant for the flood prediction.

## Objectives

- Develop a prediction model for river flood alerts in Tigre, Buenos Aires, Argentina.
- Build a dataset with climatic, tidal, and alert variables for Tigre, since there is no such thing as an existing one.
- Develop an application that allows the user to select different Machine Learning models, manipulate variables used in the training of the algorithm, and study their influence when seeing the probability that a flood alert exists or not.

## About This Repository
===

├── README.md
├── data
│   ├── raw            <- Origin data.
│   ├── preprocessed   <- Data with some transformations.
│   └── featurized     <- Final data we used for the model.
│
├── notebooks          <- Jupyter notebooks for data analysis, machine learning, and scrapping demonstrations.
│
├── tigre_flood_prediction   <- Main Folder
│   ├── core      <- Streamlit App.
│   ├── models    <- Machine Learning Models Python Scripts and in Pickle format.
│   └── process   <- feature engineering python script.
│   └── utils
        ├── scrapper  <- 2 python scripts we used to scrap weather and tide data
        ├── tigre_municipio.png       


Requirements 
===

Python version > 3.10

pandas: 2.2.1
numpy: 1.26.4
streamlit: 1.32.2
sklearn: Not found
pickle: Not found
selenium: 4.18.1
beautifulsoup: 4.12.3
requests: 2.31.0
