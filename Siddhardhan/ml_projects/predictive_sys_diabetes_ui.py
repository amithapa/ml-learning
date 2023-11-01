import numpy as np
import pickle
import streamlit as st

# loading the saved model
classifier = pickle.load(open("model/trained_model.sav", "rb"))

# creating a function for Prediction


def diabetes_prediction(input_data):
    # raw_data = (1,189,60,23,846,30.1,0.398,59)
    # raw_data = (2, 141, 58, 34, 128, 25.4, 0.699, 24)

    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    if classifier.predict(input_data_reshaped)[0] == 1:
        return "The person is diabetic"

    return "The person is not diabetic"


def main():
    # giving a title
    st.title("Daibetes Prediction App")

    pregnancies = st.text_input("Number of Pregnancies")
    glucose = st.text_input("Glucose Level")
    blood_pressure = st.text_input("Blood Pressure value")
    skin_thickness = st.text_input("Skin Thickness value")
    insulin = st.text_input("Insulin Level")
    bmi = st.text_input("BMI value")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age of the Person")

    # code for prediction
    diagonsis = ""

    # creating a button for Prediction
    if st.button("Diabetes Test Result"):
        diagonsis = diabetes_prediction(
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree_function,
                age,
            ]
        )

    st.success(diagonsis)


if __name__ == "__main__":
    main()
