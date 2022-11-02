import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run():
    # Membuat title
    st.title('Exploratory Data Analysis')

    # Membuat sub header
    st.subheader('EDA untuk Analisis Dataset FIFA 2022')

    # Membuat deskripsi
    st.write('Page by *Immanuel Yosia*')

    # Membuat garis lurus
    st.markdown('---')

    # Magic syntax
    '''
    Pada page ini, penulis akan melakukan eksplorasi data sederhana.\n
    Dataset yang digunakan adalah dataset FIFA 2022.\n
    Dataset diperoleh dari sofifa.com.
    '''

    # Show dataframe
    data = pd.read_csv('https://raw.githubusercontent.com/ardhiraka/FSDS_Guidelines/master/p1/v3/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')
    st.dataframe(data)

    # Membuat barplot
    st.write('#### Plot AttackingWorkRate')
    fig = plt.figure(figsize=(15,5))
    sns.countplot(x='AttackingWorkRate', data=data)
    st.pyplot(fig)

    # Membuat histogram
    st.write('#### Histogram of Rating')
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data['Overall'], bins=30, kde=True)
    st.pyplot(fig)

    # Membuat histogram berdasarkan input user
    st.write('#### Histogram from User Input')
    opt = st.selectbox('Pilih Column : ',('Age', 'Weight', 'Height', 'ShootingTotal'))
    fig = plt.figure(figsize=(15,5))
    sns.histplot(data[opt], bins=30, kde=True)
    st.pyplot(fig)

if __name__ == '__main__':
    run()