import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

st.write("# 1. Diagnosis and Treatment of Data Missingness ")

st.caption("Author: Amdom")

df = pd.DataFrame(pd.read_csv("airquality.CSV"))
print(df.head())

#Mariginal distribution of Ozone
fig1= plt.figure(figsize=(6,6))
ax1 = fig1.add_subplot(211)
ax1.hist(df['Ozone'], color='c') 
ax1.set_title("Ozone distribution")
plt.show()

#Joint distribution of Ozone and solar
fig2= plt.figure(figsize=(6,6))
ax2=fig2.add_subplot(211)
ax2.scatter('Solar.R', "Ozone", data=df)
ax2.set_title("Joint distribution of Ozone and solar")
ax2.set_xlabel("Solar distribution")
plt.show()


st.subheader ("""1.1 The Air quality data""")      

st.text("""              
Variables:  Ozone, Solar radiation, Wind, Temprature, Month, Day
Total observation: 153 """)
st.pyplot(fig1)

st.write("#### Joint distribution of Ozone and solar (Missingness not treated)")
st.pyplot(fig2)
st.subheader("1.2 Inspecting the missing data values")
def load_data(nrow):
    data = df
    return data[1:nrow]


data_load_state = st.text("Filtering data ...")
data=load_data(100) 
if st.checkbox('Show raw data'):
    st.write("""- Looking at the top rows, we can see 2 records
             of ozone and three of Solar are missing.""")
    st.write(data)

total_missing=df.isna().sum()
print(total_missing)
percent_missing = (total_missing/len(df))*100

df_dic = {"variable": df.columns, "Total Missing":total_missing,
          "Missing % ": percent_missing}

st.text("Bellow is the overall summary of missingness: ")
st.write(pd.DataFrame(df_dic))


st.subheader("Complete Case Analaysis:Linear Regression")
imputation = st.radio(
    "Select the imputation method",
    ('Compelete case', "mean", "Linear regression", "Stochastic", "Multiple"))


if imputation =='Compelete case':
    df_withoutNA = df.dropna()
    # print(df_withoutNA.head())
    X= np.asarray(df_withoutNA["Solar.R"])
    Y= np.asarray(df_withoutNA['Ozone'])
    X=X.reshape(-1,1)
    regr= LinearRegression().fit(X,Y)
    st.write("""the coefficient and intercept estimates are: """)
    st.write(f"Intercept: {round(regr.intercept_, 2)}",  " and "
             f"Slope: {np.round(regr.coef_, 4)}")
else:
    st.write("Coming soon.....")
    

