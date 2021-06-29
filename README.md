# sentiment_analysis_classifier_Flask_api
Flask API to predict whether a comment is positive or negative   

Data Used for Training: 

IMBD movie reviews which contains their associated binary
sentiment polarity labels

The API get request /predict
will return whether a comment is positive or negative 

Example : http://127.0.0.1:5000/predict?comment="this is good"
will return "Postive" as response 

