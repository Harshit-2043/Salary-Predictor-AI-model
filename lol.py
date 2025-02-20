import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
valued = {
    "yearsofexp": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "salary": [25000, 30000, 40000, 50000, 60000, 70000, 75000, 80000, 90000, 100000]
}

frame = pd.DataFrame(valued)

# Split dataset
x = frame[["yearsofexp"]]
y = frame["salary"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Streamlit App
st.title("Salary Predictor App")
st.write("Enter the years of experience you have")

# Fixing input field
yearexp = st.number_input("Enter the number of years of experience:", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Salary"):
    prediction = model.predict(np.array([[yearexp]]))[0]  # Fixed prediction extraction
    st.success(f"Predicted salary is: {prediction:.2f}")
