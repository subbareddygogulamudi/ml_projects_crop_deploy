import streamlit as st
import numpy as np
import pandas as pd
import pickle

with open("grid.pkl","rb") as f:
    model = pickle.load(f)

st.title("ADVANCED CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING")
st.write("Enter input feature values to predict crop type")
Nitrogen = st.number_input("Nitrogen", min_value=0.0)
Phosphorus = st.number_input("Phosphorus", min_value=0.0)
Potassium = st.number_input("Potassium", min_value=0.0)
Temperature = st.number_input("Temperature")
Humidity = st.number_input("Humidity")
pH_Value = st.number_input("pH_Value")
Rainfall = st.number_input("Rainfall")
Soil_Type=st.selectbox("Soil_Type",["Silt","Clay","Saline","Sandy","Peaty","Loamy"])
Variety=st.selectbox("Variety",["Durum","Russet","Sweet","Roma","Hard Red","Beefsteak","Co 86032","Red","Flint","Arborio","Basmati","Dent","Yukon Gold","Soft Red","Co 99004","Co 0238","Cherry","Jasmine"])
if st.button("Predict Crop"):
    input_df = pd.DataFrame([{
        'Nitrogen': Nitrogen,
        'Phosphorus': Phosphorus,
        'Potassium': Potassium,
        'Temperature': Temperature,
        'Humidity': Humidity,
        'pH_Value': pH_Value,
        'Rainfall': Rainfall,
        'Soil_Type':Soil_Type,
        'Variety':Variety }])
    prediction = model.predict(input_df)[0]
    st.success(f"🌱 Recommended Crop: {prediction}")