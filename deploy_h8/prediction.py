import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

# Load Model
with open('model_lin_reg.pkl', 'rb') as file_1:
  model_lin_reg = joblib.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = joblib.load(file_2)

with open('model_encoder.pkl', 'rb') as file_3:
  model_encoder = joblib.load(file_3)

with open('list_num_cols.txt', 'r') as file_4:
  list_num_cols = json.load(file_4)

with open('list_cat_cols.txt', 'r') as file_5:
  list_cat_cols = json.load(file_5)

def run():
    # Membuat Form
    with st.form(key='form_parameters'):
        name = st.text_input('Name', value='', help='Name of the Player')
        age = st.number_input('Age', min_value=16, max_value=60, value=25, step=1, help='Age of the Player')
        price = st.number_input('Price', min_value=0, max_value=10000000, step=1, help='Price of the Player')
        weight = st.number_input('Weight', min_value=50, max_value=120, value=70, step=1, help='Weight of the Player')
        height = st.number_input('Height', min_value=100, max_value=230, value=170, step=1, help='Height of the Player')
        st.markdown('---')

        attacking_work_rate = st.selectbox('AttackingWorkRate',('Low', 'Medium', 'High'), index=1)
        defensive_work_rate = st.selectbox('DefensiveWorkRate',('Low', 'Medium', 'High'), index=1)
        st.markdown('---')

        #pace = st.slider('Pace', 0, 100, 25)
        pace = st.number_input('Pace', min_value=0, max_value=100)
        shooting = st.number_input('Shooting', min_value=0, max_value=100)
        passing = st.number_input('Passing', min_value=0, max_value=100)
        dribbling = st.number_input('Dribbling', min_value=0, max_value=100)
        defending = st.number_input('Defending', min_value=0, max_value=100)
        physicality = st.number_input('Physicality', min_value=0, max_value=100)
        st.markdown('---')

        submitted = st.form_submit_button('Predict')

    # Membuat data inference
    data_inf = {
        'Name': name,
        'Age': age,
        'Price': price,
        'Weight': weight,
        'Height': height,
        'AttackingWorkRate': attacking_work_rate,
        'DefensiveWorkRate': defensive_work_rate,
        'PaceTotal': pace,
        'ShootingTotal': shooting,
        'PassingTotal': passing,
        'DribblingTotal': dribbling,
        'DefendingTotal': defending,
        'PhysicalityTotal': physicality
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submitted:
        # Split between Numerical Columns and Categorical Columns
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]

        # Feature Scaling and Feature Encoding
        data_inf_num_scaled = model_scaler.transform(data_inf_num)
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        # Concate Numerical Columns and Categorical Columns
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

        # Predict using Linear Regression
        y_pred_inf = model_lin_reg.predict(data_inf_final)

        st.write('## Hasil Rating = '+ str(int(y_pred_inf)))

if __name__ == '__main__':
    run()