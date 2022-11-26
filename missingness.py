import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

st.write("# Diagnosis and Treatment of Data Missingness ")
df = pd.DataFrame(pd.read_csv("airquality.CSV"))
print(df.head())

plt.hist(df['Ozone'], color='c')
plt.show()
plt.hist(df['Solar.R'])
plt.show()


def load_data(nrow):
    data = df
    return data[1:nrow]


data_load_state = st.text("Filtering data for you ...")
data=load_data(100) 

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
    
st.subheader('Bar graph showing the missing values')
msno.bar(df)