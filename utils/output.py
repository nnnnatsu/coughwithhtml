import numpy as np
import streamlit as st
import pandas as pd


def  output(prediction,predicted_index,class1,class2):
    value = None
    info = ''
    class_names = ["class1", "class2"]
    class_index = predicted_index[0]
    if class_index == 0:
        value =  class_names[0]
        info = "class1"
    elif class_index == 1:   
        value =  class_names[1]
        info = "class2"
   
    st.write(f"Predictions: {value}")
    st.write(f"Note: {info}")
    
     # Convert predictions to percentages and round up
    prediction_percentages = np.ceil(prediction * 100)
    
    # Display the percentage values for each class in a table
    data = {
        "Class [diagnose]": class_names,
        "Percentage [%]": prediction_percentages[0]
    }
    df = pd.DataFrame(data)
    st.write("Prediction Percentages:")
    st.table(df)