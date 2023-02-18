import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from fancyimpute import IterativeImputer
st.write("# Diagnosis and Treatment of Data Missingness ")

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


st.subheader ("""The Air quality data""")      

st.text("""              
Variables:  Ozone, Solar radiation, Wind, Temprature, Month, Day
Total observation: 153 """)
st.pyplot(fig1)

st.write("#### Joint distribution of Ozone and solar (Missingness not treated)")
st.pyplot(fig2)
st.subheader("Inspecting the missing data values")
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
    ('Compelete case', "MICE", "KNN means", "Stochastic", "Multiple"))

if imputation =='Compelete case':
    df_withoutNA = df.dropna()
    # print(df_withoutNA.head())
    X= np.asarray(df_withoutNA["Solar.R"])
    Y= np.asarray(df_withoutNA['Ozone'])
    X=X.reshape(-1,1)
    regr= LinearRegression().fit(X,Y)
    # SD
    y_pred = regr.predict(X)
    residuals = Y - y_pred
    # calculate the standard deviation of the residuals
    std_dev = np.std(residuals)
    # calculate the standard errors of the coefficients
    n = X.shape[0]
    se1 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    st.write("""Coef stimates of CCA: """)
    st.write(f"Intercept: {round(regr.intercept_, 2)}",  " and "
             f"Slope: {np.round(regr.coef_, 4)}")
    st.write(f"Standard error: {np.round(se1,4)}")
elif imputation =="KNN means":
    #knn = KNNImputer(n_neighbors=2, add_indicator=True)
    imputer = KNNImputer(n_neighbors=5)
    # fit and transform the imputer on the data
    df_imp = imputer.fit_transform(df)

    # convert the imputed data to a Pandas dataframe
    df_imp = pd.DataFrame(df_imp, columns=df.columns)
    
    X= np.asarray(df_imp["Solar.R"])
    Y= np.asarray(df_imp['Ozone'])
    X=X.reshape(-1,1)
    regr= LinearRegression().fit(X,Y)
    # SD
    y_pred = regr.predict(X)
    residuals = Y - y_pred
    # calculate the standard deviation of the residuals
    std_dev = np.std(residuals)
    # calculate the standard errors of the coefficients
    n = X.shape[0]
    se2 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    st.write("""Coef estimates after KNN imputation: """)
    st.write(f"Intercept: {round(regr.intercept_, 2)}",  " and "
         f"Slope: {np.round(regr.coef_, 4)}")
    st.write(f"Standard error: {np.round(se2,4)}")
    
elif imputation == "MICE":
    mice_imputer = IterativeImputer()
    df_imp = df.copy(deep=True)
    df_imp.iloc[:,:] = mice_imputer.fit_transform(df_imp)
    X= np.asarray(df_imp["Solar.R"])
    Y= np.asarray(df_imp['Ozone'])
    X=X.reshape(-1,1)
    regr= LinearRegression().fit(X,Y)
    # SD
    y_pred = regr.predict(X)
    residuals = Y - y_pred
    # calculate the standard deviation of the residuals
    std_dev = np.std(residuals)
    # calculate the standard errors of the coefficients
    n = X.shape[0]
    se3 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
    st.write("""the coefficient and intercept estimates are: """)
    st.write(f"Intercept: {round(regr.intercept_, 2)}",  " and "
         f"Slope: {np.round(regr.coef_, 4)}")
    st.write(f"Standard error: {np.round(se3,4)}")
    
else:
    st.write("Coming soon .....")
    
st.subheader("Summary table")
# CCA
df_withoutNA = df.dropna()
# print(df_withoutNA.head())
X= np.asarray(df_withoutNA["Solar.R"])
Y= np.asarray(df_withoutNA['Ozone'])
X=X.reshape(-1,1)
regr1= LinearRegression().fit(X,Y)
# SD
y_pred = regr1.predict(X)
residuals = Y - y_pred
# calculate the standard deviation of the residuals
std_dev = np.std(residuals)
# calculate the standard errors of the coefficients
n = X.shape[0]
se1 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

#KNN
imputer = KNNImputer(n_neighbors=5)
# fit and transform the imputer on the data
df_imp = imputer.fit_transform(df)

# convert the imputed data to a Pandas dataframe
df_imp = pd.DataFrame(df_imp, columns=df.columns)

X= np.asarray(df_imp["Solar.R"])
Y= np.asarray(df_imp['Ozone'])
X=X.reshape(-1,1)
regr2= LinearRegression().fit(X,Y)
# SD
y_pred = regr2.predict(X)
residuals = Y - y_pred
# calculate the standard deviation of the residuals
std_dev = np.std(residuals)
# calculate the standard errors of the coefficients
n = X.shape[0]
se2 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))

#MICE
mice_imputer = IterativeImputer()
df_imp = df.copy(deep=True)
df_imp.iloc[:,:] = mice_imputer.fit_transform(df_imp)
X= np.asarray(df_imp["Solar.R"])
Y= np.asarray(df_imp['Ozone'])
X=X.reshape(-1,1)
regr3= LinearRegression().fit(X,Y)
# SD
y_pred = regr3.predict(X)
residuals = Y - y_pred
# calculate the standard deviation of the residuals
std_dev = np.std(residuals)
# calculate the standard errors of the coefficients
n = X.shape[0]
se3 = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
st.dataframe(pd.DataFrame({"Slope":[np.round(regr1.coef_, 4),
                                np.round(regr2.coef_, 4),
                                np.round(regr3.coef_, 4)],
                           "SE": [np.around(se1,4), np.round(se2,4), np.round(se3,4)]},
                          index=pd.Index(["CCA", "KNN", "MICE"])
                                ))