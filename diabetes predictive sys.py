# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


# Loading the save model
loaded_model = pickle.load(open(r'C:\Users\PMLS\OneDrive\Desktop\Internships\Nexariza internship ML\diabetes prediction\trained_model.sav', 'rb'))
scaler = pickle.load(open(r'C:\Users\PMLS\OneDrive\Desktop\Internships\Nexariza internship ML\diabetes prediction\scaler.sav', 'rb'))

input_data = (10,168,74,0,0,38,0.537,34)

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
  print('The person is Non-Diabetic')
else:
  print('The person is Diabetic')
  