import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title= "EmployeeData Dashboard",
    page_icon= "bar_chart:",
    layout="wide"
)

# Load the dataset
file_path = 'Employee data.csv'
employeedata = pd.read_csv(file_path)

# Data Cleaning
q1 = employeedata['MonthlyIncome'].quantile(0.25)
q3 = employeedata['MonthlyIncome'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Handle outliers in MonthlyIncome
def cap_outliers(value):
    if value < lower_bound:
        return lower_bound
    elif value > upper_bound:
        return upper_bound
    else:
        return value

employeedata['MonthlyIncome'] = employeedata['MonthlyIncome'].apply(cap_outliers)

bins = [20, 30, 40, 50, 60, 70]
labels = ['20-30', '31-40', '41-50', '51-60', '61-70']
employeedata['AgeGroup'] = pd.cut(employeedata['Age'], bins=bins, labels=labels, right=False)
employeedata['Gender'] = employeedata['Gender'].astype('category')
employeedata['MaritalStatus'] = employeedata['MaritalStatus'].astype('category')

# Satisfaction Analysis
def satisfaction_analysis():
    st.header("Detailed Satisfaction Analysis")

    # Overall EnvironmentSatisfaction Distribution
    st.subheader("Overall Environment Satisfaction Distribution")
    overall_satisfaction_fig, ax = plt.subplots()
    satisfaction_counts = employeedata['EnvironmentSatisfaction'].value_counts().sort_index()
    ax.bar(satisfaction_counts.index, satisfaction_counts.values, color='teal', edgecolor='black')
    ax.set_title("Environment Satisfaction Levels")
    ax.set_xlabel("Satisfaction Level")
    ax.set_ylabel("Number of Employees")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(overall_satisfaction_fig)

    # Satisfaction by Gender
    st.subheader("Environment Satisfaction by Gender")
    gender_satisfaction_fig, ax = plt.subplots()
    satisfaction_gender = employeedata.groupby('Gender')['EnvironmentSatisfaction'].mean()
    ax.bar(satisfaction_gender.index, satisfaction_gender.values, color=['lightblue', 'salmon'], edgecolor='black')
    ax.set_title("Average Satisfaction by Gender")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Average Satisfaction Level")
    st.pyplot(gender_satisfaction_fig)

    # Recommendations for Improvement
    st.subheader("Recommendations for Improvement")
    st.write("1. **Focus on Low Satisfaction Groups**: Tailor initiatives to improve satisfaction levels, especially for employees with lower ratings.")
    st.write("2. **Gender-Specific Programs**: Develop engagement programs that address gender-specific needs based on observed trends.")
    st.write("3. **Conduct Surveys**: Use surveys to gather detailed insights into the reasons for dissatisfaction and take targeted actions.")

# Streamlit Dashboard
st.title("Employee Data Analysis Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
age_group_filter = st.sidebar.multiselect(
    "Select Age Group:",
    options=employeedata['AgeGroup'].unique(),
    default=employeedata['AgeGroup'].unique()
)
gender_filter = st.sidebar.multiselect(
    "Select Gender:",
    options=employeedata['Gender'].unique(),
    default=employeedata['Gender'].unique()
)
marital_status_filter = st.sidebar.multiselect(
    "Select Marital Status:",
    options=employeedata['MaritalStatus'].unique(),
    default=employeedata['MaritalStatus'].unique()
)

# Apply filters
filtered_data = employeedata[
    (employeedata['AgeGroup'].isin(age_group_filter)) &
    (employeedata['Gender'].isin(gender_filter)) &
    (employeedata['MaritalStatus'].isin(marital_status_filter))
]

# Display filtered data
st.header("Filtered Employee Data")
st.write(filtered_data)

# 2. Data Visualization

# Histogram of Age distribution
st.subheader("Age Distribution")
age_fig, ax = plt.subplots()
ax.hist(employeedata['Age'], bins=10, color='skyblue', edgecolor='black')
ax.set_title('Age Distribution')
ax.set_xlabel('Age')
ax.set_ylabel('Frequency')
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(age_fig)

# Box plot of MonthlyIncome by Gender
st.subheader("Monthly Income by Gender")
income_fig, ax = plt.subplots()
sns.boxplot(x='Gender', y='MonthlyIncome', data=employeedata, palette='Set2', hue='Gender', ax=ax)
ax.set_title('Monthly Income by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Monthly Income')
ax.legend(title='Gender', labels=['Female', 'Male'], loc='upper left', bbox_to_anchor=(1, 1))
st.pyplot(income_fig)

# Bar chart of EnvironmentSatisfaction levels
st.subheader("Environment Satisfaction Levels")
env_fig, ax = plt.subplots()
env_satisfaction_counts = employeedata['EnvironmentSatisfaction'].value_counts().sort_index()
ax.bar(env_satisfaction_counts.index, env_satisfaction_counts.values, color='coral', edgecolor='black')
ax.set_title('Environment Satisfaction Levels')
ax.set_xlabel('Satisfaction Level')
ax.set_ylabel('Number of Employees')
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(env_fig)

# Combined visualization showing income distribution across MaritalStatus
st.subheader("Monthly Income Distribution by Marital Status and Gender")
marital_income_fig, ax = plt.subplots()
sns.boxplot(x='MaritalStatus', y='MonthlyIncome', data=employeedata, hue='Gender', palette='Pastel1', ax=ax)
ax.set_title('Monthly Income Distribution by Marital Status and Gender')
ax.set_xlabel('Marital Status')
ax.set_ylabel('Monthly Income')
ax.legend(title='Gender', loc='upper left')
st.pyplot(marital_income_fig)

# Predictive Analysis Results
st.header("Predictive Analysis")
# Prepare data for prediction
X = employeedata[['Age', 'EnvironmentSatisfaction']]
y = employeedata['MonthlyIncome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error (MSE): {mse:.2f}")
st.write(f"R-squared (R2): {r2:.2f}")

# Satisfaction Analysis
satisfaction_analysis()

