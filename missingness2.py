import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable experimental IterativeImputer
from sklearn.impute import IterativeImputer
import io

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stRadio>label {font-size: 16px;}
    </style>
""", unsafe_allow_html=True)

# Sidebar for navigation and settings
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Analysis", "About"])

if page == "Data Analysis":
    st.write("# Diagnosis and Treatment of Data Missingness")
    st.caption("Author: Amdom | March 30, 2025")

    # Load data
    df = pd.DataFrame(pd.read_csv("airquality.CSV"))

    # Sidebar filters
    st.sidebar.subheader("Filter Data")
    month_filter = st.sidebar.multiselect("Select Months", options=sorted(df['Month'].unique()), default=sorted(df['Month'].unique()))
    filtered_df = df[df['Month'].isin(month_filter)]

    # Marginal distribution of Ozone
    st.subheader("Ozone Distribution")
    fig1 = plt.figure(figsize=(8, 5))
    plt.hist(filtered_df['Ozone'].dropna(), bins=20, color='c', edgecolor='black')
    plt.title("Ozone Distribution")
    plt.xlabel("Ozone Levels")
    plt.ylabel("Frequency")
    st.pyplot(fig1)

    # Joint distribution of Ozone and Solar
    st.subheader("Joint Distribution of Ozone and Solar")
    fig2 = plt.figure(figsize=(8, 5))
    plt.scatter(filtered_df['Solar.R'], filtered_df['Ozone'], alpha=0.5, color='purple')
    plt.title("Ozone vs. Solar Radiation")
    plt.xlabel("Solar Radiation")
    plt.ylabel("Ozone")
    st.pyplot(fig2)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig3 = plt.figure(figsize=(8, 5))
    sns.heatmap(filtered_df.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title("Correlation of Variables")
    st.pyplot(fig3)

    # Boxplot for Ozone by Month
    st.subheader("Ozone by Month")
    fig4 = plt.figure(figsize=(8, 5))
    sns.boxplot(x='Month', y='Ozone', data=filtered_df, palette='Set2')
    plt.title("Ozone Distribution by Month")
    st.pyplot(fig4)

    # Data Overview
    st.subheader("The Air Quality Data")
    st.text("Variables: Ozone, Solar Radiation, Wind, Temperature, Month, Day\nTotal Observations: 153")

    if st.checkbox('Show Raw Data'):
        st.write("- Inspect the top rows to identify missing values.")
        st.write(filtered_df.head(10))

    # Missing Data Summary
    st.subheader("Inspecting Missing Data")
    total_missing = filtered_df.isna().sum()
    percent_missing = (total_missing / len(filtered_df)) * 100
    df_dic = {"Variable": filtered_df.columns, "Total Missing": total_missing, "Missing %": percent_missing.round(2)}
    st.write(pd.DataFrame(df_dic))

    # Download missing data summary
    csv = pd.DataFrame(df_dic).to_csv(index=False)
    st.download_button("Download Missing Data Summary", csv, "missing_data_summary.csv", "text/csv")

    # Imputation and Regression
    st.subheader("Complete Case Analysis & Imputation")
    imputation = st.radio("Select Imputation Method", ('Complete Case', 'MICE', 'KNN Means', 'Stochastic', 'Multiple'))

    if imputation == 'Complete Case':
        df_clean = filtered_df.dropna()
        X = np.asarray(df_clean["Solar.R"]).reshape(-1, 1)
        Y = np.asarray(df_clean['Ozone'])
        regr = LinearRegression().fit(X, Y)
        y_pred = regr.predict(X)
        residuals = Y - y_pred
        std_dev = np.std(residuals)
        n = X.shape[0]
        se = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
        st.write(f"Intercept: {round(regr.intercept_, 2)} | Slope: {np.round(regr.coef_[0], 4)}")
        st.write(f"Standard Error: {np.round(se, 4)}")

        # Residual Plot
        fig5 = plt.figure(figsize=(8, 5))
        plt.scatter(y_pred, residuals, color='blue', alpha=0.5)
        plt.axhline(0, color='red', linestyle='--')
        plt.title("Residual Plot (Complete Case)")
        plt.xlabel("Predicted Ozone")
        plt.ylabel("Residuals")
        st.pyplot(fig5)

    elif imputation == "KNN Means":
        imputer = KNNImputer(n_neighbors=5)
        df_imp = pd.DataFrame(imputer.fit_transform(filtered_df), columns=filtered_df.columns)
        X = np.asarray(df_imp["Solar.R"]).reshape(-1, 1)
        Y = np.asarray(df_imp['Ozone'])
        regr = LinearRegression().fit(X, Y)
        y_pred = regr.predict(X)
        residuals = Y - y_pred
        std_dev = np.std(residuals)
        n = X.shape[0]
        se = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
        st.write(f"Intercept: {round(regr.intercept_, 2)} | Slope: {np.round(regr.coef_[0], 4)}")
        st.write(f"Standard Error: {np.round(se, 4)}")
        st.download_button("Download KNN Imputed Data", df_imp.to_csv(index=False), "knn_imputed_data.csv", "text/csv")

    elif imputation == "MICE":
        mice_imputer = IterativeImputer()
        df_imp = pd.DataFrame(mice_imputer.fit_transform(filtered_df), columns=filtered_df.columns)
        X = np.asarray(df_imp["Solar.R"]).reshape(-1, 1)
        Y = np.asarray(df_imp['Ozone'])
        regr = LinearRegression().fit(X, Y)
        y_pred = regr.predict(X)
        residuals = Y - y_pred
        std_dev = np.std(residuals)
        n = X.shape[0]
        se = std_dev / np.sqrt(n) * np.sqrt(np.diag(np.linalg.inv(X.T @ X)))
        st.write(f"Intercept: {round(regr.intercept_, 2)} | Slope: {np.round(regr.coef_[0], 4)}")
        st.write(f"Standard Error: {np.round(se, 4)}")
        st.download_button("Download MICE Imputed Data", df_imp.to_csv(index=False), "mice_imputed_data.csv", "text/csv")

    else:
        st.write("Coming soon...")

    # Summary Table
    st.subheader("Summary of Regression Results")
    summary_data = {
        "Method": ["CCA", "KNN", "MICE"],
        "Slope": [],
        "SE": []
    }
    for method in ["CCA", "KNN", "MICE"]:
        if method == "CCA":
            df_temp = filtered_df.dropna()
            regr = LinearRegression().fit(np.asarray(df_temp["Solar.R"]).reshape(-1, 1), df_temp['Ozone'])
            y_pred = regr.predict(np.asarray(df_temp["Solar.R"]).reshape(-1, 1))
            residuals = df_temp['Ozone'] - y_pred
            std_dev = np.std(residuals)
            se = std_dev / np.sqrt(len(df_temp)) * np.sqrt(np.diag(np.linalg.inv(np.asarray(df_temp["Solar.R"]).reshape(-1, 1).T @ np.asarray(df_temp["Solar.R"]).reshape(-1, 1))))
        elif method == "KNN":
            df_temp = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(filtered_df), columns=filtered_df.columns)
            regr = LinearRegression().fit(np.asarray(df_temp["Solar.R"]).reshape(-1, 1), df_temp['Ozone'])
            y_pred = regr.predict(np.asarray(df_temp["Solar.R"]).reshape(-1, 1))
            residuals = df_temp['Ozone'] - y_pred
            std_dev = np.std(residuals)
            se = std_dev / np.sqrt(len(df_temp)) * np.sqrt(np.diag(np.linalg.inv(np.asarray(df_temp["Solar.R"]).reshape(-1, 1).T @ np.asarray(df_temp["Solar.R"]).reshape(-1, 1))))
        else:  # MICE
            df_temp = pd.DataFrame(IterativeImputer().fit_transform(filtered_df), columns=filtered_df.columns)
            regr = LinearRegression().fit(np.asarray(df_temp["Solar.R"]).reshape(-1, 1), df_temp['Ozone'])
            y_pred = regr.predict(np.asarray(df_temp["Solar.R"]).reshape(-1, 1))
            residuals = df_temp['Ozone'] - y_pred
            std_dev = np.std(residuals)
            se = std_dev / np.sqrt(len(df_temp)) * np.sqrt(np.diag(np.linalg.inv(np.asarray(df_temp["Solar.R"]).reshape(-1, 1).T @ np.asarray(df_temp["Solar.R"]).reshape(-1, 1))))
        summary_data["Slope"].append(np.round(regr.coef_[0], 4))
        summary_data["SE"].append(np.round(se[0], 4))
    st.dataframe(pd.DataFrame(summary_data))

elif page == "About":
    st.write("# About This App")
    st.write("This app analyzes air quality data, focusing on missing data treatment and regression analysis. More features to be added soon!")