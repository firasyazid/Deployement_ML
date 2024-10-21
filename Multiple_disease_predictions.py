# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:38:21 2024

@author: firas
"""

import pickle 
import streamlit as st 
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox


diabet_model = pickle.load(open('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/trained_model.sav', 'rb'))
scaler = pickle.load(open('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/scaler.sav', 'rb'))
model_covid = joblib.load('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/svc_model.joblib')
svc = pickle.load(open('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/svcMedecineRecommandation.pkl', 'rb'))
description = pd.read_csv("C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/description.csv")
precautions = pd.read_csv("C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/precautions_df.csv")
symptoms_des = pd.read_csv("C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy//symtoms_df.csv")
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
#medical system functions
def get_predict(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Function to get description and precautions
def get_description_precautions(disease):
    desc = description[description['Disease'] == disease]['Description'].iloc[0]
    pre = precautions[precautions['Disease'] == disease].iloc[0, 1:].tolist()
    return desc, pre

#covid functions
symptom_descriptions = {
    'Breathing Problem': 'Shortness of breath or difficulty breathing.',
    'Fever': 'An abnormal rise in body temperature, usually caused by infection.',
    'Dry Cough': 'A cough that does not produce phlegm or mucus.',
    'Sore throat': 'Pain or irritation in the throat, often worsened by swallowing.',
    'Running Nose': 'Excessive production of mucus from the nasal passages.',
    'Asthma': 'A condition in which airways narrow and swell, producing extra mucus.',
    'Chronic Lung Disease': 'A long-term lung condition that impairs breathing.',
    'Headache': 'A continuous pain in the head.',
    'Heart Disease': 'A range of conditions that affect the heart.',
    'Diabetes': 'A metabolic disease in which blood sugar levels are elevated.',
    'Hyper Tension': 'High blood pressure, a condition in which the force of the blood against the artery walls is too high.',
    'Fatigue': 'A feeling of tiredness or lack of energy.',
    'Gastrointestinal': 'Related to the stomach and intestines, often causing digestive issues.',
    'Abroad travel': 'Recent travel to another country.',
    'Contact with COVID Patient': 'Direct contact with someone who has tested positive for COVID-19.',
    'Visited Public Exposed Places': 'Recent visit to places where exposure to COVID-19 is more likely, such as crowded areas.'
}

def predict_result(symptoms):
    input_data = pd.DataFrame(symptoms, index=[0])
    input_data_reshaped = input_data.values.reshape(1, -1)
    prediction = model_covid.predict(input_data_reshaped)
    return prediction










  



def predict_diabetes(data):
    std_data = scaler.transform(data)
    prediction = diabet_model.predict(std_data)
    return prediction


def preprocess_input(input_data):
    input_data_scaled = (input_data - min_train) / range_train
    return input_data_scaled


svc_model = joblib.load('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/svm_model.pkl')
min_train = np.load('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/min_train.npy')
range_train = np.load('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/range_train.npy')

#side bar  

with st.sidebar:
    selected = option_menu('Multiple Diseases prediction systems',['Medical Diagnosis','Covid Prediction','Diabetes Prediction', 'Cancer Breast Prediction','Heart Disease Prediction'],
                           icons = ['activity','activity','activity', 'person','heart-pulse'],
                           
                           default_index = 0)
    
    
# Diabetes Page

if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')

    pregnancies = st.slider('Pregnancies', 0, 17, 3)
    glucose = st.slider('Glucose', 0, 199, 117)
    blood_pressure = st.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.slider('Skin Thickness', 0, 99, 23)
    insulin = st.slider('Insulin', 0, 846, 30)
    bmi = st.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.slider('Diabetes Pedigree Function', 0.078, 3.42, 0.3725)
    age = st.slider('Age', 21, 81, 29)

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = predict_diabetes(input_data)

    # Display prediction
    if st.button('Predict'):
        if prediction[0] == 0:
            st.write('The person is not diabetic.')
        else:
            st.write('The person is diabetic.')




# Cancer Page

if(selected == 'Cancer Breast Prediction'):

    st.title ('Cancer Breast Prediction')  
    
    mean_radius = st.slider('Mean Radius', min_value=0.0, max_value=50.0, value=7.76)
    mean_texture = st.slider('Mean Texture', min_value=0.0, max_value=50.0, value=24.54)
    mean_perimeter = st.slider('Mean Perimeter', min_value=0.0, max_value=300.0, value=47.92)
    mean_area = st.slider('Mean Area', min_value=0.0, max_value=2000.0, value=181.0)
    mean_smoothness = st.slider('Mean Smoothness', min_value=0.0, max_value=1.0, value=0.05263)
    mean_compactness = st.slider('Mean Compactness', min_value=0.0, max_value=1.0, value=0.04362)
    mean_concavity = st.slider('Mean Concavity', min_value=0.0, max_value=1.0, value=0.00000)
    mean_concave_points = st.slider('Mean Concave Points', min_value=0.0, max_value=1.0, value=0.00000)
    mean_symmetry = st.slider('Mean Symmetry', min_value=0.0, max_value=1.0, value=0.1587)
    mean_fractal_dimension = st.slider('Mean Fractal Dimension', min_value=0.0, max_value=1.0, value=0.05884)
    radius_error = st.slider('Radius Error', min_value=0.0, max_value=2.0, value=0.3857)
    texture_error = st.slider('Texture Error', min_value=0.0, max_value=5.0, value=1.428)
    perimeter_error = st.slider('Perimeter Error', min_value=0.0, max_value=10.0, value=2.548)
    area_error = st.slider('Area Error', min_value=0.0, max_value=100.0, value=19.15)
    smoothness_error = st.slider('Smoothness Error', min_value=0.0, max_value=0.02, value=0.007189)
    compactness_error = st.slider('Compactness Error', min_value=0.0, max_value=0.02, value=0.004660)
    concavity_error = st.slider('Concavity Error', min_value=0.0, max_value=0.02, value=0.00000)
    concave_points_error = st.slider('Concave Points Error', min_value=0.0, max_value=0.02, value=0.00000)
    symmetry_error = st.slider('Symmetry Error', min_value=0.0, max_value=0.1, value=0.02676)
    fractal_dimension_error = st.slider('Fractal Dimension Error', min_value=0.0, max_value=0.01, value=0.002783)
    worst_radius = st.slider('Worst Radius', min_value=0.0, max_value=50.0, value=9.456)
    worst_texture = st.slider('Worst Texture', min_value=0.0, max_value=50.0, value=30.37)
    worst_perimeter = st.slider('Worst Perimeter', min_value=0.0, max_value=300.0, value=59.16)
    worst_area = st.slider('Worst Area', min_value=0.0, max_value=2000.0, value=268.6)
    worst_smoothness = st.slider('Worst Smoothness', min_value=0.0, max_value=1.0, value=0.08996)
    worst_compactness = st.slider('Worst Compactness', min_value=0.0, max_value=1.0, value=0.06444)
    worst_concavity = st.slider('Worst Concavity', min_value=0.0, max_value=1.0, value=0.0000)
    worst_concave_points = st.slider('Worst Concave Points', min_value=0.0, max_value=1.0, value=0.0000)
    worst_symmetry = st.slider('Worst Symmetry', min_value=0.0, max_value=1.0, value=0.2871)
    worst_fractal_dimension = st.slider('Worst Fractal Dimension', min_value=0.0, max_value=1.0, value=0.07039)
    # Add sliders for other features

    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'mean radius': [mean_radius],
        'mean texture': [mean_texture],
        'mean perimeter': [mean_perimeter],
        'mean area': [mean_area],
        'mean smoothness': [mean_smoothness],
        'mean compactness': [mean_compactness],
        'mean concavity': [mean_concavity],
        'mean concave points': [mean_concave_points],
        'mean symmetry': [mean_symmetry],
        'mean fractal dimension': [mean_fractal_dimension],
        'radius error': [radius_error],
        'texture error': [texture_error],
        'perimeter error': [perimeter_error],
        'area error': [area_error],
        'smoothness error': [smoothness_error],
        'compactness error': [compactness_error],
        'concavity error': [concavity_error],
        'concave points error': [concave_points_error],
        'symmetry error': [symmetry_error],
        'fractal dimension error': [fractal_dimension_error],
        'worst radius': [worst_radius],
        'worst texture': [worst_texture],
        'worst perimeter': [worst_perimeter],
        'worst area': [worst_area],
        'worst smoothness': [worst_smoothness],
        'worst compactness': [worst_compactness],
        'worst concavity': [worst_concavity],
        'worst concave points': [worst_concave_points],
        'worst symmetry': [worst_symmetry],
        'worst fractal dimension': [worst_fractal_dimension]
    })


    # Preprocess the input data
    input_data_scaled = preprocess_input(input_data)

    # Predict the result
    if st.button('Predict'):
        # Predict the result
        prediction = svc_model.predict(input_data_scaled)

        # Display the result
        if prediction[0] == 0:
            st.write('The tumor is predicted to be **malignant**.')
        else:
            st.write('The tumor is predicted to be **benign**.')
            
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')   

    age = st.slider('Age', min_value=0, max_value=120, value=55)
    sex = st.selectbox('Sex', ['Male', 'Female'], index=1)
    cp = st.slider('Chest Pain Type (cp)', min_value=0, max_value=5, value=1)
    trestbps = st.slider('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=130)
    chol = st.slider('Serum Cholesterol (chol)', min_value=0, max_value=600, value=240)
    fbs = st.slider('Fasting Blood Sugar (fbs)', min_value=0, max_value=1, value=0)
    restecg = st.slider('Resting Electrocardiographic Results (restecg)', min_value=0, max_value=2, value=1)
    thalach = st.slider('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=250, value=153)
    exang = st.slider('Exercise Induced Angina (exang)', min_value=0, max_value=1, value=0)
    oldpeak = st.slider('ST Depression Induced by Exercise Relative to Rest (oldpeak)', min_value=0.0, max_value=10.0, value=0.8)
    slope = st.slider('Slope of the Peak Exercise ST Segment (slope)', min_value=0, max_value=2, value=1)
    ca = st.slider('Number of Major Vessels Colored by Flourosopy (ca)', min_value=0, max_value=4, value=0)
    thal = st.slider('Thalassemia (thal)', min_value=0, max_value=3, value=2)
      
      # Create a DataFrame from the user input
    input_data = pd.DataFrame({
          'age': [age],
          'sex': [1 if sex == 'Male' else 0],
          'cp': [cp],
          'trestbps': [trestbps],
          'chol': [chol],
          'fbs': [fbs],
          'restecg': [restecg],
          'thalach': [thalach],
          'exang': [exang],
          'oldpeak': [oldpeak],
          'slope': [slope],
          'ca': [ca],
          'thal': [thal]
      })

      # Load the trained mode
    model = joblib.load('C:/Users/firas/OneDrive/Desktop/Machine learning/Deploy/logistic_regression_model_Heart_disease.joblib')

      # Preprocess the input data
    input_data_scaled = input_data  # No preprocessing needed for this model

      # Make predictions
    if st.button('Predict'):
          prediction = model.predict(input_data_scaled)

          # Display the prediction result
          if prediction[0] == 0:
              st.write('The patient is **not likely** to have heart disease.')
          else:
              st.write('The patient is **likely** to have heart disease.')
       
             

if selected == 'Covid Prediction':
    st.title('Covid Prediction')   
         
    breathing_problem = st.checkbox('Breathing Problem')
    if breathing_problem:
        st.write(symptom_descriptions['Breathing Problem'])

    fever = st.checkbox('Fever')
    if fever:
        st.write(symptom_descriptions['Fever'])

    dry_cough = st.checkbox('Dry Cough')
    if dry_cough:
        st.write(symptom_descriptions['Dry Cough'])

    sore_throat = st.checkbox('Sore throat')
    if sore_throat:
        st.write(symptom_descriptions['Sore throat'])

    running_nose = st.checkbox('Running Nose')
    if running_nose:
        st.write(symptom_descriptions['Running Nose'])

    asthma = st.checkbox('Asthma')
    if asthma:
        st.write(symptom_descriptions['Asthma'])

    chronic_lung_disease = st.checkbox('Chronic Lung Disease')
    if chronic_lung_disease:
        st.write(symptom_descriptions['Chronic Lung Disease'])

    headache = st.checkbox('Headache')
    if headache:
        st.write(symptom_descriptions['Headache'])

    heart_disease = st.checkbox('Heart Disease')
    if heart_disease:
        st.write(symptom_descriptions['Heart Disease'])

    diabetes = st.checkbox('Diabetes')
    if diabetes:
        st.write(symptom_descriptions['Diabetes'])

    hyper_tension = st.checkbox('Hyper Tension')
    if hyper_tension:
        st.write(symptom_descriptions['Hyper Tension'])

    fatigue = st.checkbox('Fatigue')
    if fatigue:
        st.write(symptom_descriptions['Fatigue'])

    gastrointestinal = st.checkbox('Gastrointestinal')
    if gastrointestinal:
        st.write(symptom_descriptions['Gastrointestinal'])

    abroad_travel = st.checkbox('Abroad travel')
    if abroad_travel:
        st.write(symptom_descriptions['Abroad travel'])

    contact_with_covid_patient = st.checkbox('Contact with COVID Patient')
    if contact_with_covid_patient:
        st.write(symptom_descriptions['Contact with COVID Patient'])

    visited_public_exposed_places = st.checkbox('Visited Public Exposed Places')
    if visited_public_exposed_places:
        st.write(symptom_descriptions['Visited Public Exposed Places'])

    # Convert user input to binary values
    symptoms = {
        'Breathing Problem': [1 if breathing_problem else 0],
        'Fever': [1 if fever else 0],
        'Dry Cough': [1 if dry_cough else 0],
        'Sore throat': [1 if sore_throat else 0],
        'Running Nose': [1 if running_nose else 0],
        'Asthma': [1 if asthma else 0],
        'Chronic Lung Disease': [1 if chronic_lung_disease else 0],
        'Headache': [1 if headache else 0],
        'Heart Disease': [1 if heart_disease else 0],
        'Diabetes': [1 if diabetes else 0],
        'Hyper Tension': [1 if hyper_tension else 0],
        'Fatigue': [1 if fatigue else 0],
        'Gastrointestinal': [1 if gastrointestinal else 0],
        'Abroad travel': [1 if abroad_travel else 0],
        'Contact with COVID Patient': [1 if contact_with_covid_patient else 0],
        'Visited Public Exposed Places': [1 if visited_public_exposed_places else 0]
    }

    # Make a prediction when the user clicks the button
    if st.button('Check Result'):
        # Check if no symptoms are selected
        if not any(symptoms.values()):
            st.warning('Please select at least one symptom.')
        else:
            prediction = predict_result(symptoms)
            if prediction[0] == 0:
                st.write('Based on the symptoms provided, the result is Negative.')
            else:
                st.write('Based on the symptoms provided, the result is Positive.')
      
            st.warning("Please note: While this tool can provide insights based on your symptoms, it's important to consult a doctor or healthcare professional for proper diagnosis and treatment.")
            
            
if(selected == 'Medical Diagnosis'):

    st.title ('Medical Diagnosis')  
    
    selected_symptoms = st.multiselect('Select your symptoms (at least 4 symtoms):', list(symptoms_dict.keys()))

    if selected_symptoms:
         predicted_disease = get_predict(selected_symptoms)
         st.subheader('Predicted Disease:')
         st.write(predicted_disease)

         # Get description and precautions
         desc, pre = get_description_precautions(predicted_disease)
         st.subheader('Description:')
         st.write(desc)
         st.subheader('Precautions:')
         for i, p in enumerate(pre, start=1):
             st.write(f'{i}: {p}')
             
         st.warning("Please note: While this tool can provide insights based on your symptoms, it's important to consult a doctor or healthcare professional for proper diagnosis and treatment.")

    