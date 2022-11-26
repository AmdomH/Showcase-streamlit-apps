import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno

st.write("# Diagnosis and Treatment of Data Missingness ")
df = pd.DataFrame(pd.read_csv("airquality.CSV"))
print(df.head())

fig = plt.figure(figsize=(6,6))
plt.subplot(211)
plt.hist(df['Ozone'], color='c') 
plt.show()

plt.subplot(212)
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
st.write ("### Histogram of Ozone and Solar")
st.pyplot(fig)
st.subheader('Bar graph showing the missing values')
msno.bar(df)