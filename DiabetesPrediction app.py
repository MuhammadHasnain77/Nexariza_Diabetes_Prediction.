# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:10:15 2024

@author: PMYLS
"""

import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open(r'C:\Users\PMLS\OneDrive\Desktop\Internships\Nexariza internship ML\diabetes prediction\trained_model.sav', 'rb'))
scaler = pickle.load(open(r'C:\Users\PMLS\OneDrive\Desktop\Internships\Nexariza internship ML\diabetes prediction\scaler.sav', 'rb'))

# create a function for prediction

def diabetes_prediction(input_data):

    # Changing input data to numpy array
    input_data_numpy_array = np.asarray(input_data)

    #reshape array as we are predicting for one instance
    input_data_reshape = input_data_numpy_array.reshape(1, -1)

    # standardize the input data
    std_data = scaler.transform(input_data_reshape)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0] == 0):
        
      return'The person is Non-Diabetic'
      
    else:
        
      return'The person is Diabetic'
      
def main():
    
    # Title giving
    st.title('Diabetes Predictive System')
    
    # Giving the input data
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure= st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of a Person')

    # code for Prediction
    diagnosis = ''
    
    # creating a clickable button for a Prediction
    if st.button('Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)


if __name__ == '__main__':
    main()